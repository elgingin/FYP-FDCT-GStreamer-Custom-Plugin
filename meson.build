# Find required dependencies
cuda = find_program('/usr/local/cuda/bin/nvcc', required: true)
gstreamer = dependency('gstreamer-1.0', required: true)
gst_video = dependency('gstreamer-video-1.0', required: true)
gst_rtp = dependency('gstreamer-rtp-1.0', required: true)

# Include directories
cuda_include = include_directories('/usr/local/cuda/include')
header_dir = include_directories('include') 

# Source files
encoder_src = 'plugin.c'
decoder_src = 'plugin.c'
rtp_payloader_src = 'fdctcustompay.c'
rtp_depayloader_src = 'fdctcustomdepay.c'

# CUDA Compilation Targets
cuda_args = ['-Xcompiler', '-fPIC', '-O3']


compress_target = custom_target(
    'compress',
    input: '/home/jerry/gst-plugins-bad/ext/fastdct/compression/compress.cu',
    output: 'compress.o',
    command: [cuda, '-c', '@INPUT@', '-o', '@OUTPUT@', '-I/usr/local/cuda/include', '-I/home/jerry/gst-plugins-bad/ext/cuscodec/include', '-O3', '-g', '-Xcompiler', '-fPIC'],
    build_by_default: true
)


decompress_target = custom_target(
    'decompress',
    input: '/home/jerry/gst-plugins-bad/ext/fastdct/decompression/decompress.cu',
    output: 'decompress.o',
    command: [cuda, '-c', '@INPUT@', '-o', '@OUTPUT@', '-I/usr/local/cuda/include', '-I/home/jerry/gst-plugins-bad/ext/cuscodec/include', '-O3', '-g', '-Xcompiler', '-fPIC'],
    build_by_default: true
)


# Libraries and linking
cuda_lib_path = '/usr/local/cuda/lib64'
cuda_libs = ['-lcufft', '-lcuda', '-lcudart', '-lstdc++']

# Build a shared GStreamer plugin library (Fix - Add compress.o & decompress.o)
gst_fastdct_lib = shared_library(
    'gstfastdct',
    [encoder_src, decoder_src, 'compressionpart.c', 'decompressionpart.c', compress_target, decompress_target],  # ✅ Fixed
    include_directories: [cuda_include, header_dir],
    dependencies: [gstreamer, gst_video, gst_rtp],
    link_args: ['-L' + cuda_lib_path] + cuda_libs,
    install: true,
    install_dir: get_option('libdir') / 'gstreamer-1.0'
)

# Build encoder plugin (Fix - Add compress.o)
encoder_plugin = shared_library(
    'fdctgstcustomencoder',
    [compress_target, 'compressionpart.c'],  # ✅ Keep compress_target
    include_directories: [cuda_include, header_dir],
    dependencies: [gstreamer, gst_video],
    link_args: ['-L' + cuda_lib_path] + cuda_libs,
    install: true,
    install_dir: get_option('libdir') / 'gstreamer-1.0'
)

# Build decoder plugin (Fix - Add decompress.o)
decoder_plugin = shared_library(
    'fdctgstcustomdecoder',
    [decompress_target, 'decompressionpart.c'],  # ✅ Keep decompress_target
    include_directories: [cuda_include, header_dir],
    dependencies: [gstreamer, gst_video],
    link_args: ['-L' + cuda_lib_path] + cuda_libs,
    install: true,
    install_dir: get_option('libdir') / 'gstreamer-1.0'
)

# Build RTP payloader plugin
rtp_payloader_plugin = shared_library(
    'fdctcustompay',
    [rtp_payloader_src],
    include_directories: [header_dir],
    dependencies: [gstreamer, gst_video, gst_rtp],
    install: true,
    install_dir: get_option('libdir') / 'gstreamer-1.0'
)

# Build RTP depayloader plugin
rtp_depayloader_plugin = shared_library(
    'fdctcustomdepay',
    [rtp_depayloader_src],
    include_directories: [header_dir],
    dependencies: [gstreamer, gst_video, gst_rtp],
    install: true,
    install_dir: get_option('libdir') / 'gstreamer-1.0'
)

summary({
    'FastDCT GStreamer Plugin': gst_fastdct_lib.full_path(),
    'FastDCT Encoder Plugin': encoder_plugin.full_path(),
    'FastDCT Decoder Plugin': decoder_plugin.full_path(),
    'FastDCT RTP Payloader Plugin': rtp_payloader_plugin.full_path(),
    'FastDCT RTP Depayloader Plugin': rtp_depayloader_plugin.full_path(),
    'FastDCT CUDA Include': '/usr/local/cuda/include',
    'FastDCT Header Directory': 'include'
})
