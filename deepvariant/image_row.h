/*
 * Copyright 2024 Google LLC.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef LEARNING_GENOMICS_DEEPVARIANT_IMAGE_ROW_H_
#define LEARNING_GENOMICS_DEEPVARIANT_IMAGE_ROW_H_

#include <vector>

#include "deepvariant/protos/deepvariant.pb.h"
#include "absl/log/log.h"

namespace learning {
namespace genomics {
namespace deepvariant {

struct ImageRow {
  int width;
  int num_channels;
  std::vector<DeepVariantChannelEnum> channel_enums;
  std::vector<unsigned char> flat_data;  // [num_channels * width], channel-major

  inline unsigned char* channel(int ch) {
    return flat_data.data() + ch * width;
  }
  inline const unsigned char* channel(int ch) const {
    return flat_data.data() + ch * width;
  }

  int Width() const;
  explicit ImageRow(int width, int num_channels);
  bool operator==(const ImageRow& other) const {
    if (channel_enums != other.channel_enums) {
      LOG(INFO) << "ImageRow channel_enums mismatch";
    }
    if (flat_data != other.flat_data) {
      LOG(INFO) << "ImageRow flat_data mismatch";
    }
    return width == other.width && num_channels == other.num_channels &&
           channel_enums == other.channel_enums &&
           flat_data == other.flat_data;
  }
};

}  // namespace deepvariant
}  // namespace genomics
}  // namespace learning

#endif  // LEARNING_GENOMICS_DEEPVARIANT_IMAGE_ROW_H_
