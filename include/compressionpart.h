#ifndef COMPRESSIONPART_H
#define COMPRESSIONPART_H

#include <gst/gst.h>
#include "compress.h"  // Add this line

G_BEGIN_DECLS

#define GST_TYPE_FDCT_COMPRESS (gst_fdct_compress_get_type())
G_DECLARE_FINAL_TYPE(GstFdctCompress, gst_fdct_compress, GST, FDCT_COMPRESS, GstElement)

// Function to perform the compression transform
GstFlowReturn gst_fdct_compress_transform(GstBuffer *inbuf, GstBuffer **outbuf);

G_END_DECLS

#endif /* COMPRESSIONPART_H */
