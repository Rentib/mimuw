use sha2::{Digest, Sha256};
use std::path::PathBuf;
use tokio::fs::{self, File, OpenOptions};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
// You can add here other imports from std or crates listed in Cargo.toml.

// You can add any private types, structs, consts, functions, methods, etc., you need.
// As always, you should not modify the public interfaces.
struct Storage {
    root_storage_dir: PathBuf,
}

#[async_trait::async_trait]
pub trait StableStorage: Send + Sync {
    /// Stores `value` under `key`.
    ///
    /// Detailed requirements are specified in the description of the assignment.
    async fn put(&mut self, key: &str, value: &[u8]) -> Result<(), String>;

    /// Retrieves value stored under `key`.
    ///
    /// Detailed requirements are specified in the description of the assignment.
    async fn get(&self, key: &str) -> Option<Vec<u8>>;

    /// Removes `key` and the value stored under it.
    ///
    /// Detailed requirements are specified in the description of the assignment.
    async fn remove(&mut self, key: &str) -> bool;
}

/// Creates a new instance of stable storage.
pub async fn build_stable_storage(root_storage_dir: PathBuf) -> Box<dyn StableStorage> {
    Box::new(Storage { root_storage_dir })
}

impl Storage {
    fn key_path(&self, key: &str) -> Result<(PathBuf, PathBuf), String> {
        if key.len() > 255 {
            return Err("Key too long".to_string());
        }

        let mut path = self.root_storage_dir.clone();
        let key = format!("{:x}", Sha256::digest(key.as_bytes()));
        path.push(key);
        let tmp_path = path.with_extension("tmp");
        Ok((path, tmp_path))
    }

    async fn sync_tmp(&self, dst: PathBuf, tmp: PathBuf) {
        if let Ok(mut tmp_file) = File::open(tmp.clone()).await {
            let mut tmp_data = Vec::new();
            tmp_file.read_to_end(&mut tmp_data).await.unwrap();

            if tmp_data.len() >= 32 {
                let data_len = tmp_data.len() - 32;
                let (tmp_data, tmp_checksum) = tmp_data.split_at(data_len);
                // Recalculate the SHA-256 checksum of the data portion
                let mut hasher = Sha256::new();
                hasher.update(tmp_data);
                let calculated_checksum = hasher.finalize();
                if calculated_checksum.as_slice() == tmp_checksum {
                    // Proceed to Step 4: Write data from tmp to dst
                    let mut dst_file = OpenOptions::new()
                        .write(true)
                        .create(true)
                        .truncate(true)
                        .open(dst.clone())
                        .await
                        .unwrap();
                    dst_file.write_all(tmp_data).await.unwrap();
                    dst_file.sync_data().await.unwrap();
                    // Step 5: Sync dst file
                    dst_file.sync_data().await.unwrap();
                    // Step 6: Sync dst file's directory
                    let dst_dir = dst
                        .parent()
                        .expect("dst file should have a parent directory");
                    let dst_dir_file = File::open(dst_dir).await.unwrap();
                    dst_dir_file.sync_data().await.unwrap();
                    // 7. Remove dstdir/tmpfile.
                    fs::remove_file(tmp).await.unwrap();
                }
            }
        }
    }
}

#[async_trait::async_trait]
impl StableStorage for Storage {
    async fn put(&mut self, key: &str, value: &[u8]) -> Result<(), String> {
        if value.len() > 65536 {
            return Err("Value too big".to_string());
        }
        let (dst, tmp) = self.key_path(key)?;
        self.sync_tmp(dst.clone(), tmp.clone()).await;

        // 1. Write the data with a checksum (e.g., CRC32) to a temporary file dstdir/tmpfile.
        let mut tmp_file = File::create(tmp.clone()).await.unwrap();
        let mut hasher = Sha256::new();
        hasher.update(value);
        let checksum = hasher.finalize();
        tmp_file.write_all(value).await.unwrap();
        tmp_file.write_all(&checksum).await.unwrap();
        // 2. Call the POSIX fsyncdata function on dstdir/tmpfile to ensure the data is actually
        //    transferred to a disk device (in Rust, one can use the tokio::fs::File::sync_data()
        //    method).
        tmp_file.sync_data().await.unwrap();
        // 3. Call fsyncdata on dstdir to transfer the data of the modified directory to the disk
        //    device. (Again, in Rust, one can use the tokio::fs::File::sync_data() method. Even
        //    though the struct is called File, here it can be used for directories as well, for
        //    example: tokio::fs::File::open("dir").await.unwrap().sync_data().await.unwrap()).
        let tmp_dir = tmp.parent().expect("tmp has no parent directory");
        let tmp_dir_file = File::open(tmp_dir).await.unwrap();
        tmp_dir_file.sync_data().await.unwrap();
        // 4. Write the data (without the checksum) to dstdir/dstfile.
        let mut dst_file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(dst.clone())
            .await
            .unwrap();
        dst_file.write_all(value).await.unwrap();
        // 5. Call fsyncdata on dstdir/dstfile.
        dst_file.sync_data().await.unwrap();
        // 6. Call fsyncdata on dstdir (only necessary if dstfile did not exist before the previous step).
        let dst_dir = dst.parent().expect("tmp has no parent directory");
        let dst_dir_file = File::open(dst_dir).await.unwrap();
        dst_dir_file.sync_data().await.unwrap();
        // 7. Remove dstdir/tmpfile.
        fs::remove_file(tmp).await.unwrap();
        // 8. Call fsyncdata on dstdir.
        dst_dir_file.sync_data().await.unwrap();

        Ok(())
    }

    async fn get(&self, key: &str) -> Option<Vec<u8>> {
        let (dst, tmp) = self.key_path(key).ok()?;
        self.sync_tmp(dst.clone(), tmp.clone()).await;
        tokio::fs::read(&dst).await.ok()
    }

    async fn remove(&mut self, key: &str) -> bool {
        let (dst, tmp) = self.key_path(key).ok().unwrap();
        self.sync_tmp(dst.clone(), tmp.clone()).await;
        tokio::fs::remove_file(&dst).await.is_ok()
    }
}
