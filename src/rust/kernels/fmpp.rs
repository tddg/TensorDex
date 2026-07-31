//! FM++ codec — reduction-oriented delta compression, extended from FM-Delta [71].
//!
//! Thin FFI over the FM-Delta arithmetic residual coder (vendored prebuilt at
//! `third_party/fmdelta/libfmdelta.a`; built only under the `fmpp` cargo
//! feature). Treats the tensor as a 1-D HALF array (`type_ = 2`), matching the
//! encode path that produced the published `fratio` / `fbytes_out` columns — so
//! `compress_fmpp_rust` reproduces them bit-for-bit.  The matching
//! `decompress_fmpp_rust` binding uses the read side of the same library so
//! the AE can verify a real base + delta -> target byte-exact round trip.

use pyo3::prelude::*;

const FMD_TYPE_HALF: i32 = 2;

#[repr(C)]
struct FMD {
    type_: i32,
    nx: i32,
    ny: i32,
    nz: i32,
    nf: i32,
}

extern "C" {
    fn fmd_write_to_buffer(buffer: *mut u8, size: usize) -> *mut FMD;
    fn fmd_write_header(fmd: *mut FMD) -> i32;
    fn fmd_write(fmd: *mut FMD, base_data: *const u8, finetuned_data: *const u8) -> usize;
    fn fmd_write_close(fmd: *mut FMD);

    fn fmd_read_from_buffer(buffer: *const u8) -> *mut FMD;
    fn fmd_read_header(fmd: *mut FMD) -> i32;
    fn fmd_read(fmd: *mut FMD, data: *mut u8, base_data: *const u8) -> usize;
    fn fmd_read_close(fmd: *mut FMD);
}

fn validate_half_input(data_len: usize, item_size: usize) -> Result<usize, String> {
    if item_size != 2 {
        return Err(format!(
            "fmpp: only 2-byte fp16/bf16 elements are supported, got item_size={item_size}"
        ));
    }
    if data_len == 0 || data_len % item_size != 0 {
        return Err(format!(
            "fmpp: byte length {data_len} must be a non-zero multiple of {item_size}"
        ));
    }
    let n_elements = data_len / item_size;
    if n_elements > i32::MAX as usize {
        return Err("fmpp: tensor has too many elements for the FM-Delta header".into());
    }
    Ok(n_elements)
}

/// Compress one tensor with FM++.  This non-Python entry point is public so
/// the artifact's native Rayon benchmark can schedule independent tensors
/// across cores without going through (or being serialized by) the GIL.
pub fn compress_fmpp(target: &[u8], base: &[u8], item_size: usize) -> Result<Vec<u8>, String> {
    if base.len() != target.len() {
        return Err("fmpp: base and target must have equal byte length".into());
    }
    let n_elements = validate_half_input(base.len(), item_size)?;
    let buf_size = base
        .len()
        .checked_add(28 + 1024)
        .ok_or_else(|| "fmpp: output buffer size overflow".to_string())?;
    let mut buffer = vec![0u8; buf_size];

    let out = unsafe {
        let p = fmd_write_to_buffer(buffer.as_mut_ptr(), buf_size);
        if p.is_null() {
            return Err("fmpp: fmd_write_to_buffer returned null".into());
        }
        (*p).type_ = FMD_TYPE_HALF;
        (*p).nx = n_elements as i32;
        (*p).ny = 1;
        (*p).nz = 1;
        (*p).nf = 1;
        if fmd_write_header(p) == 0 {
            fmd_write_close(p);
            return Err("fmpp: fmd_write_header failed".into());
        }
        let n = fmd_write(p, base.as_ptr(), target.as_ptr());
        fmd_write_close(p);
        n
    };
    if out == 0 || out > buffer.len() {
        return Err("fmpp: fmd_write failed or exceeded its output buffer".into());
    }
    buffer.truncate(out);
    Ok(buffer)
}

/// Decompress one FM++ tensor delta.  See [`compress_fmpp`] for why the
/// native entry point is exposed in addition to the PyO3 wrapper.
pub fn decompress_fmpp(
    compressed: &[u8],
    base: &[u8],
    item_size: usize,
) -> Result<Vec<u8>, String> {
    let expected_elements = validate_half_input(base.len(), item_size)?;
    if compressed.is_empty() {
        return Err("fmpp: compressed input is empty".into());
    }

    let mut output = vec![0u8; base.len()];
    let read = unsafe {
        let p = fmd_read_from_buffer(compressed.as_ptr());
        if p.is_null() {
            return Err("fmpp: fmd_read_from_buffer returned null".into());
        }
        if fmd_read_header(p) == 0 {
            fmd_read_close(p);
            return Err("fmpp: fmd_read_header failed".into());
        }

        let dims = ((*p).nx, (*p).ny, (*p).nz, (*p).nf);
        let header_elements =
            [dims.0, dims.1, dims.2, dims.3]
                .into_iter()
                .try_fold(1usize, |acc, dim| {
                    if dim <= 0 {
                        None
                    } else {
                        acc.checked_mul(dim as usize)
                    }
                });
        if (*p).type_ != FMD_TYPE_HALF || header_elements != Some(expected_elements) {
            fmd_read_close(p);
            return Err(format!(
                "fmpp: header does not match the base tensor \
                 (type={}, dims={}x{}x{}x{}, expected {} half elements)",
                (*p).type_,
                dims.0,
                dims.1,
                dims.2,
                dims.3,
                expected_elements
            ));
        }

        let n = fmd_read(p, output.as_mut_ptr(), base.as_ptr());
        fmd_read_close(p);
        n
    };
    if read == 0 {
        return Err("fmpp: fmd_read failed".into());
    }
    Ok(output)
}

/// Encode `target` as an FM++ delta against `base`; returns the compressed bytes.
/// `item_size` is the element width in bytes (2 for bf16/fp16).
#[pyfunction]
#[pyo3(signature = (target, base, item_size=2))]
pub fn compress_fmpp_rust(
    py: Python,
    target: &[u8],
    base: &[u8],
    item_size: usize,
) -> PyResult<PyObject> {
    let buffer = compress_fmpp(target, base, item_size)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    Ok(pyo3::types::PyBytes::new(py, &buffer).to_object(py))
}

/// Decode an FM++ delta against `base`; returns the original target bytes.
#[pyfunction]
#[pyo3(signature = (compressed, base, item_size=2))]
pub fn decompress_fmpp_rust(
    py: Python,
    compressed: &[u8],
    base: &[u8],
    item_size: usize,
) -> PyResult<PyObject> {
    let output = decompress_fmpp(compressed, base, item_size)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    Ok(pyo3::types::PyBytes::new(py, &output).to_object(py))
}

#[cfg(test)]
mod tests {
    use super::{compress_fmpp, decompress_fmpp};

    #[test]
    fn round_trip_half_bytes() {
        let base: Vec<u8> = (0..8192u32)
            .flat_map(|i| (i as u16).to_le_bytes())
            .collect();
        let target: Vec<u8> = (0..8192u32)
            .flat_map(|i| ((i as u16).wrapping_add((i % 17) as u16)).to_le_bytes())
            .collect();
        let compressed = compress_fmpp(&target, &base, 2).unwrap();
        let decoded = decompress_fmpp(&compressed, &base, 2).unwrap();
        assert_eq!(decoded, target);
    }
}
