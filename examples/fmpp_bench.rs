//! FM++ aggregate throughput benchmark (in-memory, CPU-only).
//!
//! FM-Delta's arithmetic coder is single-threaded for one tensor.  The paper
//! benchmark scheduled independent tensor pairs on Rayon, which is also the
//! natural unit of parallelism in TensorDex.  This program reproduces that
//! path and verifies every decoded tensor byte-for-byte before timing it.
//!
//! Synthetic smoke test (no download):
//!   cargo run --release --features fmpp --example fmpp_bench
//!
//! Real Qwen2.5-7B Base -> Instruct pair:
//!   cargo run --release --features fmpp --example fmpp_bench -- \
//!     --base-model BASE_DIR --target-model TARGET_DIR

use rayon::prelude::*;
use std::collections::HashMap;
use std::time::Instant;
use tensordex_ops::kernels::fmpp::{compress_fmpp, decompress_fmpp};

const ITEM_SIZE: usize = 2;
const SYNTH_PAIR_BYTES: usize = 4 * 1024 * 1024;

struct Args {
    base_model: Option<String>,
    target_model: Option<String>,
    size_mb: usize,
    threads: Option<usize>,
    warmup: usize,
    iters: usize,
}

fn parse_args() -> Args {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut args = Args {
        base_model: None,
        target_model: None,
        size_mb: 512,
        threads: None,
        warmup: 1,
        iters: 3,
    };
    let mut i = 0;
    while i < raw.len() {
        let value = |flag: &str, pos: usize| {
            raw.get(pos + 1)
                .unwrap_or_else(|| panic!("{flag} requires a value"))
                .clone()
        };
        match raw[i].as_str() {
            "--base-model" => {
                args.base_model = Some(value("--base-model", i));
                i += 2;
            }
            "--target-model" => {
                args.target_model = Some(value("--target-model", i));
                i += 2;
            }
            "--size-mb" => {
                args.size_mb = value("--size-mb", i).parse().expect("invalid --size-mb");
                i += 2;
            }
            "--threads" => {
                args.threads = Some(value("--threads", i).parse().expect("invalid --threads"));
                i += 2;
            }
            "--warmup" => {
                args.warmup = value("--warmup", i).parse().expect("invalid --warmup");
                i += 2;
            }
            "--iters" => {
                args.iters = value("--iters", i).parse().expect("invalid --iters");
                i += 2;
            }
            other => panic!("unknown argument: {other}"),
        }
    }
    assert_eq!(
        args.base_model.is_some(),
        args.target_model.is_some(),
        "--base-model and --target-model must be supplied together"
    );
    assert!(args.size_mb > 0, "--size-mb must be positive");
    assert!(args.iters > 0, "--iters must be positive");
    if let Some(n) = args.threads {
        assert!(n > 0, "--threads must be positive");
    }
    args
}

fn load_model(dir: &str) -> HashMap<String, Vec<u8>> {
    let mut out = HashMap::new();
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read {dir}: {e}"))
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no .safetensors files in {dir}");
    for path in files {
        let raw =
            std::fs::read(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let tensors = safetensors::SafeTensors::deserialize(&raw)
            .unwrap_or_else(|e| panic!("invalid {}: {e}", path.display()));
        for (name, view) in tensors.tensors() {
            use safetensors::Dtype::{BF16, F16, I16, U16};
            if matches!(view.dtype(), BF16 | F16 | I16 | U16) {
                out.insert(name.to_string(), view.data().to_vec());
            }
        }
    }
    out
}

fn real_pairs(base_dir: &str, target_dir: &str) -> Vec<(Vec<u8>, Vec<u8>)> {
    eprintln!("loading {base_dir} ...");
    let mut base = load_model(base_dir);
    eprintln!("loading {target_dir} ...");
    let target = load_model(target_dir);
    let mut names: Vec<_> = target.keys().cloned().collect();
    names.sort();
    names
        .into_iter()
        .filter_map(|name| {
            let t = target.get(&name)?;
            let b = base.remove(&name)?;
            (t.len() == b.len() && !t.is_empty()).then(|| (t.clone(), b))
        })
        .collect()
}

fn synthetic_pairs(total_bytes: usize) -> Vec<(Vec<u8>, Vec<u8>)> {
    let total_bytes = total_bytes.next_multiple_of(ITEM_SIZE);
    let mut pairs = Vec::new();
    let mut remaining = total_bytes;
    let mut seed = 0x1234_5678u32;
    while remaining > 0 {
        let n = remaining.min(SYNTH_PAIR_BYTES).next_multiple_of(ITEM_SIZE);
        let mut base = vec![0u8; n];
        let mut target = vec![0u8; n];
        for i in (0..n).step_by(2) {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let b = (seed >> 8) as u16;
            let d = (seed % 33) as i16 - 16;
            let t = (b as i16).wrapping_add(d) as u16;
            base[i..i + 2].copy_from_slice(&b.to_le_bytes());
            target[i..i + 2].copy_from_slice(&t.to_le_bytes());
        }
        pairs.push((target, base));
        remaining = remaining.saturating_sub(n);
    }
    pairs
}

fn throughput<F>(label: &str, bytes: usize, warmup: usize, iters: usize, f: F)
where
    F: Fn() + Sync,
{
    for _ in 0..warmup {
        f();
    }
    let mut elapsed = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        f();
        elapsed.push(start.elapsed().as_secs_f64());
    }
    let avg = elapsed.iter().sum::<f64>() / iters as f64;
    let best = elapsed.iter().copied().fold(f64::INFINITY, f64::min);
    let gb = bytes as f64 / 1e9;
    println!(
        "  {label:<11} avg {:6.2} GB/s   peak {:6.2} GB/s   ({avg:.3}s avg)",
        gb / avg,
        gb / best
    );
}

fn main() {
    let args = parse_args();
    if let Some(threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .expect("failed to configure Rayon thread pool");
    }

    let (pairs, source) = match (&args.base_model, &args.target_model) {
        (Some(base), Some(target)) => (real_pairs(base, target), "real model pair"),
        _ => (
            synthetic_pairs(args.size_mb * 1024 * 1024),
            "synthetic tensor pairs",
        ),
    };
    assert!(!pairs.is_empty(), "no compatible 2-byte tensor pairs found");
    let bytes: usize = pairs.iter().map(|(t, _)| t.len()).sum();
    println!(
        "FM++ aggregate throughput - {source} ({} MB, {} pairs, {} threads)",
        bytes >> 20,
        pairs.len(),
        rayon::current_num_threads()
    );

    // Materialize deltas once for both integrity verification and decode timing.
    let compressed: Vec<Vec<u8>> = pairs
        .par_iter()
        .map(|(target, base)| compress_fmpp(target, base, ITEM_SIZE).expect("FM++ encode failed"))
        .collect();
    let compressed_bytes: usize = compressed.iter().map(Vec::len).sum();
    let exact = pairs
        .par_iter()
        .zip(compressed.par_iter())
        .all(|((target, base), delta)| {
            decompress_fmpp(delta, base, ITEM_SIZE).is_ok_and(|decoded| decoded == *target)
        });
    assert!(exact, "FM++ round-trip mismatch");
    println!(
        "  reduction   {:.3}x ({:.1}% saved)   round-trip byte-exact",
        compressed_bytes as f64 / bytes as f64,
        100.0 * (1.0 - compressed_bytes as f64 / bytes as f64)
    );

    throughput("compress", bytes, args.warmup, args.iters, || {
        pairs.par_iter().for_each(|(target, base)| {
            let _ = compress_fmpp(target, base, ITEM_SIZE).expect("FM++ encode failed");
        });
    });
    throughput("decompress", bytes, args.warmup, args.iters, || {
        pairs
            .par_iter()
            .zip(compressed.par_iter())
            .for_each(|((_, base), delta)| {
                let _ = decompress_fmpp(delta, base, ITEM_SIZE).expect("FM++ decode failed");
            });
    });
    println!("RESULT: PASS - every decoded tensor is byte-for-byte identical");
}
