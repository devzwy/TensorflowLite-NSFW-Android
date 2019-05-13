/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.example.tensorflowlite_nsfw_android.tflite;

import android.app.Activity;

import java.io.IOException;

/**
 * This TensorFlowLite classifier works with the float MobileNet model.
 */
public class ClassifierFloatMobileNet extends Classifier {


    /**
     * Initializes a {@code ClassifierFloatMobileNet}.
     *
     * @param activity
     */
    public ClassifierFloatMobileNet(Activity activity, Device device, int numThreads)
            throws IOException {
        super(activity, device, numThreads);
    }

    @Override
    public int getImageSizeX() {
        return 224;
    }

    @Override
    public int getImageSizeY() {
        return 224;
    }

    @Override
    protected String getModelPath() {
        // you can download this file from
        // see build.gradle for where to obtain this file. It should be auto
        // downloaded into assets.
        return "nsfw.tflite";
    }

    @Override
    protected int getNumBytesPerChannel() {
        return 4; // Float.SIZE / Byte.SIZE;
    }

    @Override
    protected void addPixelValue(int pixelValue) {

//        int red = (pixelValue >> 16) & 0xFF;
//        int green = (pixelValue >> 8) & 0xFF;
//        int blue = (pixelValue >> 0) & 0xFF;
//        int out = (blue << 16) | (green << 8) | (red << 0);
//        imgData.putFloat(Float.parseFloat(String.valueOf(out)));
    }

}
