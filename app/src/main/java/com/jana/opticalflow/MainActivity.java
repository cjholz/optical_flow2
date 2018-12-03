package com.jana.opticalflow;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.WindowManager;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import static org.opencv.video.Video.OPTFLOW_FARNEBACK_GAUSSIAN;
import static org.opencv.video.Video.OPTFLOW_USE_INITIAL_FLOW;
import static org.opencv.video.Video.calcOpticalFlowFarneback;
import static org.opencv.video.Video.calcOpticalFlowPyrLK;


public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2{
    private CameraBridgeViewBase cameraView;
    private Mat previousFrame;
    private MatOfPoint2f prevPts;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        OpenCVLoader.initDebug();
        cameraView = findViewById(R.id.camera_view);
        cameraView.setCvCameraViewListener(this);
        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA},
                    1);
        }
        else {
            cameraView.enableView();
        }
        prevPts = new MatOfPoint2f();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String permissions[], int[] grantResults) {
        switch (requestCode) {
            case 1: {
                // If request is cancelled, the result arrays are empty.
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    cameraView.enableView();
                }
                return;
            }

            // other 'case' lines to check for other
            // permissions this app might request.
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat gray = inputFrame.gray();
        Mat rgb = inputFrame.rgba();
        MatOfPoint2f nextPts = new MatOfPoint2f();
        MatOfByte status = new MatOfByte();
        MatOfFloat err = new MatOfFloat();
        if (previousFrame != null) {
            calcOpticalFlowPyrLK(previousFrame, gray, prevPts, nextPts, status, err);
        }

        for (int i=0; i < nextPts.toList().size(); i++) {
            Imgproc.line(rgb, prevPts.toList().get(i), nextPts.toList().get(i), new Scalar(255,0,0), 2);
        }
        previousFrame = inputFrame.gray();
        MatOfPoint points = new MatOfPoint();
        Imgproc.goodFeaturesToTrack(previousFrame, points, 100, .01, 50);
        prevPts.fromArray(points.toArray());
        return rgb;
    }
}
