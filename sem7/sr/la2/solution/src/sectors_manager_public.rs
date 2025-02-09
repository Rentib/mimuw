use crate::{util::N_WORKERS, SectorIdx, SectorVec};
use sha2::{Digest, Sha256};
use std::{
    collections::{hash_map::Entry, HashMap},
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::{
    fs::{read, remove_file, File},
    io::AsyncWriteExt,
    sync::RwLock,
};

#[async_trait::async_trait]
pub trait SectorsManager: Send + Sync {
    /// Returns 4096 bytes of sector data by index.
    async fn read_data(&self, idx: SectorIdx) -> SectorVec;

    /// Returns timestamp and write rank of the process which has saved this data.
    /// Timestamps and ranks are relevant for atomic register algorithm, and are described
    /// there.
    async fn read_metadata(&self, idx: SectorIdx) -> (u64, u8);

    /// Writes a new data, along with timestamp and write rank to some sector.
    async fn write(&self, idx: SectorIdx, sector: &(SectorVec, u64, u8));
}

struct InternalSectorsManager {
    metadata: Vec<RwLock<HashMap<SectorIdx, (u64, u8)>>>,
    path: PathBuf,
}

impl InternalSectorsManager {
    async fn new(path: PathBuf) -> Self {
        let metadata = Self::recover(path.clone()).await;
        Self { metadata, path }
    }

    fn make_path(path: &Path, idx: SectorIdx, timestamp: u64, write_rank: u8) -> PathBuf {
        path.join(format!("{}_{}_{}", idx, timestamp, write_rank))
    }

    fn make_tmp_path(path: &Path, idx: SectorIdx) -> PathBuf {
        path.join(format!("tmp_{}", idx))
    }

    async fn recover(path: PathBuf) -> Vec<RwLock<HashMap<SectorIdx, (u64, u8)>>> {
        let mut metadata = (0..N_WORKERS).map(|_| HashMap::new()).collect::<Vec<_>>();

        let mut dir: tokio::fs::ReadDir = tokio::fs::read_dir(&path).await.unwrap();
        while let Some(file) = dir.next_entry().await.unwrap() {
            let filename = file.file_name();
            let parts: Vec<_> = filename.to_str().unwrap().split('_').collect();
            if parts.len() != 4 || parts[0] != "tmp" {
                continue;
            }
            let content = read(file.path()).await.unwrap();
            if content.len() >= 32 {
                let (data, checksum) = content.split_at(content.len() - 32);
                let mut hasher = Sha256::new();
                hasher.update(data);
                let calculated_checksum = hasher.finalize();
                if calculated_checksum.as_slice() == checksum {
                    let path = path.clone();
                    let sector_idx = parts[1].parse::<SectorIdx>().unwrap();
                    let timestamp = parts[2].parse::<u64>().unwrap();
                    let write_rank = parts[3].parse::<u8>().unwrap();

                    let mut dst_file =
                        File::create(Self::make_path(&path, sector_idx, timestamp, write_rank))
                            .await
                            .unwrap();
                    dst_file.write_all(data).await.unwrap();
                    dst_file.sync_data().await.unwrap();
                    drop(dst_file);
                    File::open(path).await.unwrap().sync_data().await.unwrap();
                    remove_file(file.path()).await.unwrap();
                }
            }
            remove_file(file.path()).await.unwrap();
            continue;
        }

        let mut dir: tokio::fs::ReadDir = tokio::fs::read_dir(&path).await.unwrap();
        while let Some(file) = dir.next_entry().await.unwrap() {
            let filename = file.file_name();
            let parts: Vec<_> = filename.to_str().unwrap().split('_').collect();
            if parts.len() != 3 {
                remove_file(file.path()).await.unwrap();
                continue;
            }

            let sector_idx = parts[0].parse::<SectorIdx>().unwrap();
            let timestamp = parts[1].parse::<u64>().unwrap();
            let write_rank = parts[2].parse::<u8>().unwrap();

            match metadata[(sector_idx % N_WORKERS) as usize].entry(sector_idx) {
                Entry::Occupied(mut entry) => {
                    let (old_timestamp, old_write_rank) = entry.get();
                    if (*old_timestamp, *old_write_rank) < (timestamp, write_rank) {
                        remove_file(Self::make_path(
                            &path,
                            sector_idx,
                            *old_timestamp,
                            *old_write_rank,
                        ))
                        .await
                        .unwrap();
                        *entry.get_mut() = (timestamp, write_rank);
                    } else {
                        remove_file(file.path()).await.unwrap();
                    }
                }
                Entry::Vacant(entry) => {
                    entry.insert((timestamp, write_rank));
                }
            }
        }

        metadata.into_iter().map(|m| RwLock::new(m)).collect()
    }
}

#[async_trait::async_trait]
impl SectorsManager for InternalSectorsManager {
    async fn read_data(&self, idx: SectorIdx) -> SectorVec {
        let metadata = self.metadata[(idx % N_WORKERS) as usize].read().await;
        match metadata.get(&idx) {
            Some((timestamp, write_rank)) => SectorVec(
                read(InternalSectorsManager::make_path(
                    &self.path,
                    idx,
                    *timestamp,
                    *write_rank,
                ))
                .await
                .unwrap(),
            ),
            None => SectorVec(vec![0u8; 4096]),
        }
    }

    async fn read_metadata(&self, idx: SectorIdx) -> (u64, u8) {
        let metadata = self.metadata[(idx % N_WORKERS) as usize].read().await;
        *metadata.get(&idx).unwrap_or(&(0, 0))
    }

    async fn write(&self, idx: SectorIdx, sector: &(SectorVec, u64, u8)) {
        let (SectorVec(data), timestamp, write_rank) = sector;

        let dst = InternalSectorsManager::make_path(&self.path, idx, *timestamp, *write_rank);
        let tmp = InternalSectorsManager::make_tmp_path(&self.path, idx);

        let mut tmp_file = File::create(&tmp).await.unwrap();
        let mut hasher = Sha256::new();
        hasher.update(data);
        let checksum = hasher.finalize();
        tmp_file.write_all(data).await.unwrap();
        tmp_file.write_all(&checksum).await.unwrap();
        tmp_file.sync_data().await.unwrap();
        drop(tmp_file);
        File::open(&self.path)
            .await
            .unwrap()
            .sync_data()
            .await
            .unwrap();

        let mut dst_file = File::create(&dst).await.unwrap();
        dst_file.write_all(data).await.unwrap();
        dst_file.sync_data().await.unwrap();
        drop(dst_file);
        File::open(&self.path)
            .await
            .unwrap()
            .sync_data()
            .await
            .unwrap();
        remove_file(&tmp).await.unwrap();
        File::open(&self.path)
            .await
            .unwrap()
            .sync_data()
            .await
            .unwrap();

        // NOTE: here it's important to remove old file, otherwise we leak memory
        let mut metadata = self.metadata[(idx % N_WORKERS) as usize].write().await;
        if let Some((old_timestamp, old_write_rank)) =
            metadata.insert(idx, (*timestamp, *write_rank))
        {
            remove_file(InternalSectorsManager::make_path(
                &self.path,
                idx,
                old_timestamp,
                old_write_rank,
            ))
            .await
            .unwrap();
            File::open(&self.path)
                .await
                .unwrap()
                .sync_data()
                .await
                .unwrap();
        }
    }
}

/// Path parameter points to a directory to which this method has exclusive access.
pub async fn build_sectors_manager(path: PathBuf) -> Arc<dyn SectorsManager> {
    Arc::new(InternalSectorsManager::new(path).await)
}
