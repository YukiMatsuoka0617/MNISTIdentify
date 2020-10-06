package com.example.mnistidentify;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Path;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.ViewTreeObserver;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity implements View.OnClickListener {

    TextView textView;

    Button buttonReset;
    Button buttonIdenfify;

    ImageView iv;
    PaintView paintView;

    Bitmap bitmap = null;
    Module module = null;

    Bitmap drawBitmap;
    Canvas drawCanvas;

    private ArrayList<Integer> classnames = new ArrayList<Integer>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        findView();
        button();

        try {
            module = Module.load(assetFilePath(this, "CNNModel.pt"));
        }
        catch (IOException e) {
            Log.e("PytorchHelloWorld", "Error reading assets", e);
            finish();
        }

        for(int i = 0; i < 10;i++){
            classnames.add(i);
        }
    }

    void findView(){
        buttonReset = findViewById(R.id.resetButton);
        buttonIdenfify = findViewById(R.id.identifyButton);
        textView = findViewById(R.id.textView);
        iv = findViewById(R.id.imageView);
        paintView = findViewById(R.id.view);
    }

    void button(){
        buttonReset.setOnClickListener(this);
        buttonIdenfify.setOnClickListener(this);
    }

    void recognition(Bitmap image){
        // preparing input tensor
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(image,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);

        // running the model
        final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

        // getting tensor content as java array of floats
        final float[] scores = outputTensor.getDataAsFloatArray();

        // searching for the index with maximum score
        float maxScore = -Float.MAX_VALUE;
        int maxScoreIdx = -1;
        for (int i = 0; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                maxScoreIdx = i;
            }
        }

        String className = String.valueOf(classnames.get(maxScoreIdx));

        // showing className on UI
        textView.setText("Class:" + className);
    }

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.identifyButton:
                drawBitmap = Bitmap.createBitmap(paintView.getWidth(),paintView.getHeight(), Bitmap.Config.ARGB_8888);
                drawCanvas = new Canvas(drawBitmap);
                drawCanvas.drawRGB(0,0,0);
                for (Path path : PaintView.pathList) {
                    drawCanvas.drawPath(path, PaintView.paint);
                }
                bitmap = Bitmap.createScaledBitmap(drawBitmap,28,28,true);
                iv.setImageBitmap(bitmap);

                recognition(bitmap);
                break;
            case R.id.resetButton:
                paintView.clear(); //画面をクリアに
                break;
        }
    }


}