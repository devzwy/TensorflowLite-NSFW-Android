package com.example.tensorflowlite_nsfw_android

import android.annotation.TargetApi
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import com.example.tensorflowlite_nsfw_android.tflite.Classifier
import com.zwy.xlog.XLog
import java.util.*

class MainActivity : AppCompatActivity() {
    var classifier: Classifier? = null
    @TargetApi(Build.VERSION_CODES.JELLY_BEAN_MR2)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)


        if (classifier == null)
            classifier = Classifier.create(this, Classifier.Device.CPU, 10)
        XLog.d("TensorflowLite", "开始识别")
        val startTime = Date().time


        val nsfwBean = classifier?.run(
            Bitmap.createScaledBitmap(
                BitmapFactory.decodeStream(getResources().getAssets().open("aaa.png")),
                224,
                224,
                false
            )
        )

        XLog.d("TensorflowLite", "识别成功：耗时${Date().time - startTime} ms,nsfw:${nsfwBean}")
    }

}
