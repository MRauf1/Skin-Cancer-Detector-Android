package com.example.m_rau.skincancerdetector;

import android.Manifest;
import android.annotation.TargetApi;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.v4.content.ContextCompat;
import android.support.v4.content.FileProvider;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends AppCompatActivity {

    private static final int START_CAMERA_ACTIVITY = 0;
    private static final int OPEN_GALLERY_ACTIVITY = 1;
    private static final int EXTERNAL_STORAGE_REQUEST_RESULT = 1;
    private TensorFlowInferenceInterface inferenceInterface;
    private ImageView imageViewCapturedPhoto;
    private String imageLocation;

    //Variables about the Skin Cancer Detector model
    private static final String MODEL_PATH = "file:///android_asset/skin_cancer_detector_model.pb";
    private static final String INPUT_NODE = "conv2d_1_input";
    private static final String OUTPUT_NODE = "dense_3/Sigmoid";

    //Dimensions of the input
    private static final int TARGET_WIDTH = 224;
    private static final int TARGET_HEIGHT = 224;

    //Target size required for the Skin Cancer Detector input
    private static final int[] INPUT_SIZE = {1, TARGET_WIDTH, TARGET_HEIGHT, 3};

    static {
        System.loadLibrary("tensorflow_inference");
    }



    //Initialize and set up the whole screen and variables
    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageViewCapturedPhoto = findViewById(R.id.imageViewPhoto);
        Button loadImage = findViewById(R.id.loadimage);

        //Click Listener for the "Load Image" buttong
        //Whenever the button is clicked, redirect the user to the Gallery to select the image
        loadImage.setOnClickListener(new Button.OnClickListener() {

            @Override
            public void onClick(View arg0) {

                //Access the Gallery
                Intent intent = new Intent(Intent.ACTION_PICK,
                        android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);

                //Run after the image has been selected
                startActivityForResult(intent, OPEN_GALLERY_ACTIVITY);

            }});

        //Prepare the Tensorflow Inference to run the model
        inferenceInterface = new TensorFlowInferenceInterface();
        inferenceInterface.initializeTensorFlow(getAssets(), MODEL_PATH);

    }



    //Request/Check permissions necessary for taking the photo
    public void checkPermissions(View view) {

        //Proceed if the permission is granted
        if(ContextCompat.checkSelfPermission(this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {

            takePhoto();

        } else {

            //Explain why the permission is needed
            if(shouldShowRequestPermissionRationale(Manifest.permission.WRITE_EXTERNAL_STORAGE)) {
                Toast.makeText(this,
                        "Permission required to save the image.", Toast.LENGTH_SHORT).show();
            }

            requestPermissions(new String[] {Manifest.permission.WRITE_EXTERNAL_STORAGE}, EXTERNAL_STORAGE_REQUEST_RESULT);

        }

    }



    //Request the permission when the feature is used for the first time
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {

        //Proceed if the permission is granted
        if (requestCode == EXTERNAL_STORAGE_REQUEST_RESULT) {

            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {

                takePhoto();

            } else {

                Toast.makeText(this,
                        "External storage permission was denied.", Toast.LENGTH_SHORT).show();

            }

        } else {

            super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        }

    }



    //Access the camera so that the user can take a photo
    public void takePhoto() {

        //Open the camera
        Intent cameraIntent = new Intent();
        cameraIntent.setAction(MediaStore.ACTION_IMAGE_CAPTURE);

        File image = null;

        //Attempt to save the image
        try {
            image = createImage();
        } catch(IOException e) {
            e.printStackTrace();
        }

        cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT, FileProvider.getUriForFile(this,
                "com.example.m_rau.skincancerdetector.provider", image));

        //Run once the picture has been taken
        startActivityForResult(cameraIntent, START_CAMERA_ACTIVITY);

    }



    //Rotate the inputted image to the correct orientation
    public Bitmap rotateImage(Bitmap bitmap) {

        ExifInterface exifInterface = null;

        //Attempt to create the ExifInterface for the image
        try {
            exifInterface = new ExifInterface(imageLocation);
        } catch (IOException e) {
            e.printStackTrace();
        }

        //Get the orientation of the image
        int orientation = exifInterface.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_UNDEFINED);

        Matrix matrix = new Matrix();

        //Rotate the matrix for the image accordingly
        switch(orientation) {
            case ExifInterface.ORIENTATION_ROTATE_90:
                matrix.setRotate(90);
                break;
            case ExifInterface.ORIENTATION_ROTATE_180:
                matrix.setRotate(180);
                break;
            case ExifInterface.ORIENTATION_ROTATE_270:
                matrix.setRotate(270);
                break;
        }

        //Rotate the bitmap and display it
        Bitmap rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        imageViewCapturedPhoto.setImageBitmap(rotatedBitmap);

        return rotatedBitmap;

    }



    //Output the prediction given by running the image through the Skin Cancer Detector
    public void predict(Bitmap bitmap) {

        //Format the input
        float[] pixels = new float[TARGET_WIDTH * TARGET_HEIGHT * 3];
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, TARGET_WIDTH, TARGET_HEIGHT, false);

        //Input all the pixel values of the image into the array
        for(int i = 0; i < TARGET_WIDTH; i++) {

            for(int j = 0; j < TARGET_HEIGHT; j++) {

                int pixel = resizedBitmap.getPixel(i, j);

                pixels[(i + j * TARGET_WIDTH) * 3] = ((float) Color.red(pixel)) / 255;
                pixels[(i + j * TARGET_WIDTH) * 3 + 1] = ((float) Color.green(pixel)) / 255;
                pixels[(i + j * TARGET_WIDTH) * 3 + 2] = ((float) Color.blue(pixel)) / 255;

            }

        }

        //Run the Skin Cancer Detector with the inputted image's pixels
        inferenceInterface.fillNodeFloat(INPUT_NODE, INPUT_SIZE, pixels);
        inferenceInterface.runInference(new String[] {OUTPUT_NODE});

        //Read the prediction by the model
        float[] prediction = {0};
        inferenceInterface.readNodeFloat(OUTPUT_NODE, prediction);

        //Reformat the prediction
        String predictionText = prediction[0] < 0.5 ? "Benign" : "Malignant";

        //Display the prediction
        TextView predictionValue = findViewById(R.id.prediction);
        predictionValue.setText(predictionText);

    }



    //Activate the predicting function once an image from the gallery is selected or photo is taken
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {

        //Give the prediction on the taken photo or the selected photo from the gallery
        if(requestCode == START_CAMERA_ACTIVITY && resultCode == RESULT_OK) {

            //Retrieve and format the image
            Bitmap photoBitmap = BitmapFactory.decodeFile(imageLocation);
            Bitmap bitmap = rotateImage(photoBitmap);

            //Output the prediction
            predict(bitmap);


        } else if(requestCode == OPEN_GALLERY_ACTIVITY && resultCode == RESULT_OK) {

            Uri targetUri = data.getData();
            Bitmap bitmap;

            //Display the image and output the prediction
            try {

                bitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(targetUri));
                imageViewCapturedPhoto.setImageBitmap(bitmap);
                predict(bitmap);

            } catch (FileNotFoundException e) {

                //Print out the error message
                e.printStackTrace();

            }

        }

    }



    //Save the taken image on the device and return the file
    public File createImage() throws IOException {

        //Name of the file
        String time = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String name = "CancerImage_" + time;

        //Create the file with the assigned name
        File directory = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
        File image = File.createTempFile(name, ".jpg", directory);

        //Get the absolute path of the image
        imageLocation = image.getAbsolutePath();

        return  image;

    }

}
