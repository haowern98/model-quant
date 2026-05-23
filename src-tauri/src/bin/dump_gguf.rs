use std::env;
use std::fs::File;
use std::io::Read;

fn read_string(buf: &[u8], offset: &mut usize) -> String {
    let len = u64::from_le_bytes(buf[*offset..*offset+8].try_into().unwrap()) as usize;
    *offset += 8;
    let s = String::from_utf8_lossy(&buf[*offset..*offset+len]).to_string();
    *offset += len;
    s
}

fn read_u32(buf: &[u8], offset: &mut usize) -> u32 {
    let v = u32::from_le_bytes(buf[*offset..*offset+4].try_into().unwrap());
    *offset += 4;
    v
}

fn read_u64(buf: &[u8], offset: &mut usize) -> u64 {
    let v = u64::from_le_bytes(buf[*offset..*offset+8].try_into().unwrap());
    *offset += 8;
    v
}

fn skip_value(buf: &[u8], offset: &mut usize, value_type: u32) {
    match value_type {
        0 | 1 | 7 => *offset += 1,
        2 | 3 => *offset += 2,
        4 | 5 | 6 => *offset += 4,
        10 | 11 | 12 => *offset += 8,
        8 => { let _ = read_string(buf, offset); }
        9 => {
            let elem_type = read_u32(buf, offset);
            let count = read_u64(buf, offset) as usize;
            for _ in 0..count {
                skip_value(buf, offset, elem_type);
            }
        }
        _ => {}
    }
}

fn main() {
    let path = env::args().nth(1).expect("usage: dump_gguf <file>");
    let mut file = File::open(&path).unwrap();
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();

    let mut offset = 0usize;
    println!("File size: {} bytes", buf.len());

    let magic = read_u32(&buf, &mut offset);
    println!("\nMagic: 0x{:08X}", magic);

    let version = read_u32(&buf, &mut offset);
    println!("Version: {}", version);

    let tensor_count = read_u64(&buf, &mut offset) as usize;
    println!("Tensor count: {}", tensor_count);

    let metadata_count = read_u64(&buf, &mut offset) as usize;
    println!("Metadata KV count: {}", metadata_count);
    println!("Header end offset: {}", offset);

    println!("\n--- Metadata ({}) ---", metadata_count);
    for i in 0..metadata_count {
        let key = read_string(&buf, &mut offset);
        let value_type = read_u32(&buf, &mut offset);
        let type_name = ["u8","i8","u16","i16","u32","i32","f32","bool","string","array","u64","i64","f64"].get(value_type as usize).unwrap_or(&"?");
        match value_type {
            0..=7 | 10..=12 => { skip_value(&buf, &mut offset, value_type); }
            8 => { let val = read_string(&buf, &mut offset); }
            9 => { skip_value(&buf, &mut offset, value_type); }
            _ => {}
        }
        if i < 5 {
            println!("  [{}] key=\"{}\" type={}", i, key, type_name);
        }
    }
    println!("After metadata, offset: {}", offset);

    println!("\n--- Tensor Info (first 5 of {}) ---", tensor_count);
    for i in 0..tensor_count.min(5) {
        let name = read_string(&buf, &mut offset);
        let n_dims = read_u32(&buf, &mut offset) as usize;
        print!("  [{}] name=\"{}...\" dims={} [", i, &name[..name.len().min(50)], n_dims);
        for d in 0..n_dims {
            let dim = read_u64(&buf, &mut offset);
            print!("{}", dim);
            if d + 1 < n_dims { print!(", "); }
        }
        let ggml_type = read_u32(&buf, &mut offset);
        let tensor_offset_u64 = read_u64(&buf, &mut offset);
        print!("] type={} data_offset={}", ggml_type, tensor_offset_u64);

        // Check what bytes come after
        if i + 1 < tensor_count {
            let peek = &buf[offset..offset+8];
            println!(" next_bytes={:02x?}", peek);
        } else {
            println!();
        }
    }
    println!("After {} tensors, offset: {}", tensor_count.min(5), offset);
    println!("Bytes remaining: {}", buf.len() - offset);
}
