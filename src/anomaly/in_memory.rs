use std::collections::HashMap;
use std::sync::RwLock;

use crate::anomaly::embedding::AnomalyEmbedding;
use crate::anomaly::traits::{AnomalyError, AnomalyIndex};

/// Default capacity for each tenant's FIFO sliding window ring buffer (20,000 vectors).
pub const DEFAULT_RING_CAPACITY: usize = 20_000;

/// Contiguous circular ring buffer storing up to `capacity` embeddings.
///
/// Absorbs natural production semantic drift by evicting the oldest sample
/// once capacity is exceeded ($O(1)$ amortized insertion).
#[derive(Clone, Debug)]
pub struct FifoRingBuffer {
    capacity: usize,
    data: Vec<AnomalyEmbedding>,
    head: usize,
    count: usize,
}

impl FifoRingBuffer {
    /// Create a new circular ring buffer with the specified maximum capacity.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Capacity must be greater than zero");
        Self {
            capacity,
            data: Vec::with_capacity(capacity),
            head: 0,
            count: 0,
        }
    }

    /// Insert an embedding into the ring buffer.
    ///
    /// If capacity has not yet been reached, appends the vector.
    /// If capacity is full, overwrites the oldest element at `head` ($O(1)$ eviction).
    pub fn insert(&mut self, embedding: AnomalyEmbedding) {
        if self.data.len() < self.capacity {
            self.data.push(embedding);
            self.count += 1;
        } else {
            self.data[self.head] = embedding;
            self.head = (self.head + 1) % self.capacity;
            self.count = self.capacity;
        }
    }

    /// Number of active vectors currently stored.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the buffer has zero vectors.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Maximum capacity of the ring buffer.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clears the ring buffer.
    pub fn clear(&mut self) {
        self.data.clear();
        self.head = 0;
        self.count = 0;
    }

    /// Returns a slice of all currently stored embeddings.
    pub fn as_slice(&self) -> &[AnomalyEmbedding] {
        &self.data
    }

    /// Computes the exact Euclidean distances from `query` to all stored vectors,
    /// returning the `k` smallest distances sorted ascending.
    pub fn k_nearest_distances(&self, query: &AnomalyEmbedding, k: usize) -> Vec<f32> {
        if self.data.is_empty() || k == 0 {
            return Vec::new();
        }

        let mut distances: Vec<f32> = self
            .data
            .iter()
            .map(|target| query.euclidean_distance(target))
            .collect();

        if distances.len() <= k {
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            distances
        } else {
            distances.select_nth_unstable_by(k - 1, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut top_k = distances[..k].to_vec();
            top_k.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            top_k
        }
    }
}

/// Thread-safe in-memory vector store implementing `AnomalyIndex`.
///
/// Manages isolated tenant namespaces via `FifoRingBuffer` instances, providing
/// deterministic sub-millisecond k-NN distance computation on local CPU.
pub struct InMemoryRingIndex {
    namespaces: RwLock<HashMap<String, FifoRingBuffer>>,
    default_capacity: usize,
}

impl InMemoryRingIndex {
    /// Create an `InMemoryRingIndex` with standard 20,000 capacity per namespace.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_RING_CAPACITY)
    }

    /// Create an `InMemoryRingIndex` with custom per-tenant capacity.
    pub fn with_capacity(default_capacity: usize) -> Self {
        Self {
            namespaces: RwLock::new(HashMap::new()),
            default_capacity,
        }
    }

    /// Initialize a namespace if it does not already exist.
    pub fn ensure_namespace(&self, namespace: &str) {
        let mut map = self.namespaces.write().unwrap();
        map.entry(namespace.to_string())
            .or_insert_with(|| FifoRingBuffer::new(self.default_capacity));
    }
}

impl Default for InMemoryRingIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl AnomalyIndex for InMemoryRingIndex {
    fn search_knn(
        &self,
        namespace: &str,
        query: &AnomalyEmbedding,
        k: usize,
    ) -> Result<Vec<f32>, AnomalyError> {
        let map = self.namespaces.read().unwrap();
        match map.get(namespace) {
            Some(buffer) => Ok(buffer.k_nearest_distances(query, k)),
            None => Ok(Vec::new()),
        }
    }

    fn insert_benign(
        &self,
        namespace: &str,
        vector: AnomalyEmbedding,
    ) -> Result<(), AnomalyError> {
        let mut map = self.namespaces.write().unwrap();
        let buffer = map
            .entry(namespace.to_string())
            .or_insert_with(|| FifoRingBuffer::new(self.default_capacity));
        buffer.insert(vector);
        Ok(())
    }

    fn sample_count(&self, namespace: &str) -> usize {
        let map = self.namespaces.read().unwrap();
        map.get(namespace).map(|b| b.len()).unwrap_or(0)
    }

    fn get_all(&self, namespace: &str) -> Result<Vec<AnomalyEmbedding>, AnomalyError> {
        let map = self.namespaces.read().unwrap();
        match map.get(namespace) {
            Some(buffer) => Ok(buffer.as_slice().to_vec()),
            None => Ok(Vec::new()),
        }
    }

    fn clear(&self, namespace: &str) -> Result<(), AnomalyError> {
        let mut map = self.namespaces.write().unwrap();
        if let Some(buffer) = map.get_mut(namespace) {
            buffer.clear();
        }
        Ok(())
    }
}
