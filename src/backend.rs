//! Runtime backend selection for the binaries.
//!
//! Every [`BackendKind`] variant always parses; choosing one whose cargo feature was not
//! compiled in yields [`BackendKind::not_compiled`] instead of silently hiding the flag.

use std::fmt;

pub use burn_flex::FlexDevice;

/// Default CPU backend.
pub type CpuBackend = burn_flex::Flex<f32, i32>;

#[cfg(feature = "wgpu")]
pub use burn::backend::wgpu::WgpuDevice;
/// WebGPU backend (Vulkan on Linux).
#[cfg(feature = "wgpu")]
pub type GpuWgpu = burn::backend::Wgpu;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum BackendKind {
    #[default]
    Cpu,
    Wgpu,
}

impl BackendKind {
    pub fn name(self) -> &'static str {
        match self {
            BackendKind::Cpu => "cpu",
            BackendKind::Wgpu => "wgpu",
        }
    }

    /// Error for a backend whose cargo feature is missing from this build.
    pub fn not_compiled(self) -> Box<dyn std::error::Error> {
        let feature = match self {
            BackendKind::Cpu => "default",
            BackendKind::Wgpu => "wgpu",
        };
        format!(
            "backend `{}` is not compiled into this binary; rebuild with `--features {feature}`",
            self.name()
        )
        .into()
    }

    /// Feature-cache key: everything that changes the cached features. GPU features differ
    /// numerically from CPU ones (the encoder amplifies float noise), so each backend gets
    /// its own cache.
    pub fn cache_key(self, encoder_sha: &str, tokenizer_sha: &str, max_seq_len: usize) -> String {
        format!(
            "{}-tok{}-enc{}-len{max_seq_len}-{}",
            &encoder_sha[..encoder_sha.len().min(16)],
            &tokenizer_sha[..tokenizer_sha.len().min(12)],
            crate::encoding::ENCODING_VERSION,
            self.name()
        )
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
