package com.jana.opticalflow;

import android.Manifest;
import android.content.pm.PackageManager;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.WindowManager;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;

import static org.opencv.video.Video.calcOpticalFlowPyrLK;


public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2{
    private CameraBridgeViewBase cameraView;
    private Mat previousFrame;
    private MatOfPoint2f prevPts;
    private long prevFrameTime;
    private ArrayList<Double> xvels;
    private ArrayList<Double> yvels;
    //private ArrayList<Double> xcenters;
    //private ArrayList<Double> ycenters;


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
        xvels = new ArrayList<>();
        yvels = new ArrayList<>();
        //xcenters = new ArrayList<>();
        //ycenters = new ArrayList<>();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String permissions[], int[] grantResults) {
        switch (requestCode) {
            case 1: {
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    cameraView.enableView();
                }
                return;
            }
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
        long time = System.currentTimeMillis();
        double timeDiff = (time - prevFrameTime)/1000.;
        prevFrameTime = time;
        Mat gray = inputFrame.gray();
        Mat rgb = inputFrame.rgba();
        MatOfPoint2f nextPts = new MatOfPoint2f();
        MatOfByte status = new MatOfByte();
        MatOfFloat err = new MatOfFloat();
        if (previousFrame != null) {
            calcOpticalFlowPyrLK(previousFrame, gray, prevPts, nextPts, status, err, new Size(3, 3), 2);
        }

        double totalx = 0;
        double totaly = 0;
        double totalvelx = 0;
        double totalvely = 0;
        double totaldisplacex = 0;
        double totaldisplacey = 0;
        int numFound = 0;
        for (int i=0; i < nextPts.toList().size(); i++) {
            if (status.toList().get(i) == 1) {
                totalx += nextPts.toArray()[i].x;
                totaly += nextPts.toArray()[i].y;
                Point displacement = prevPts.toArray()[i].subtract(nextPts.toArray()[i]);
                totalvelx += displacement.x/timeDiff;
                totalvely += displacement.y/timeDiff;
                totaldisplacex += displacement.x;
                totaldisplacey += displacement.y;
                numFound++;
            }
        }
        double displacex = totaldisplacex/numFound;
        double displacey = totaldisplacey/numFound;
        double centerx = totalx/numFound;
        double centery = totaly/numFound;
        double xvelp = totalvelx/numFound;
        double yvelp = totalvely/numFound;
        double xvel = (3./rgb.cols()) * xvelp;
        double yvel = (2./rgb.rows()) * yvelp;
        if (xvels.size() >= 5) {
            xvels.remove(0);
            yvels.remove(0);
            //xcenters.remove(0);
            //ycenters.remove(0);
        }
        xvels.add(xvel);
        yvels.add(yvel);
        //xcenters.add(centerx);
        //ycenters.add(centery);
        xvel = 0;
        yvel = 0;
        //centerx = 0;
        //centery = 0;
        for(int i=0; i<xvels.size(); i++){
            xvel += xvels.get(i);
            yvel += yvels.get(i);
            //centerx += xcenters.get(i);
            //centery += ycenters.get(i);
        }
        xvel = xvel/5.;
        yvel = yvel/5.;
        //centerx = centerx/5.;
        //centery = centery/5.;

        double magn = Math.sqrt(Math.pow(xvel, 2) + Math.pow(yvel, 2));
        //TODO:Smoothing
        if (magn > 0.05) {
            Imgproc.arrowedLine(rgb, new Point(centerx, centery), new Point(centerx + displacex, centery + displacey), new Scalar(255, 0, 0), 10);
        }
        else {
            xvel = 0;
            yvel = 0;
        }
        Imgproc.putText(rgb, "XVEL" + String.valueOf(Math.round(xvel)) + "m/s", new Point(30, 50), Core.FONT_HERSHEY_PLAIN, 5.0, new Scalar(0, 0, 255));
        Imgproc.putText(rgb, "YVEL" + String.valueOf(Math.round(yvel)) + "m/s", new Point(30, 100), Core.FONT_HERSHEY_PLAIN, 5.0, new Scalar(0, 0, 255));


        previousFrame = inputFrame.gray();
        MatOfPoint points = new MatOfPoint();
        Imgproc.goodFeaturesToTrack(previousFrame, points, 10, .01, 50);
        prevPts.fromArray(points.toArray());
        return rgb;
    }
}
