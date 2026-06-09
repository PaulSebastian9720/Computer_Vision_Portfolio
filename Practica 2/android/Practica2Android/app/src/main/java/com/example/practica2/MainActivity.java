package com.example.practica2;

import androidx.appcompat.app.AppCompatActivity;
import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.view.SurfaceView;
import android.widget.ArrayAdapter;
import android.widget.SeekBar;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.CheckBox;
import android.widget.AdapterView;
import android.view.View;
import android.util.Log;

import androidx.annotation.NonNull;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import java.util.Locale;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    static {
        System.loadLibrary("practica2");
    }

    private CameraBridgeViewBase cameraView;
    private TextView statusText;
    
    private TextView alphaBlueText, sMinText, vMinText, hMinText, hMaxText, gaussText, speckleText, kernelText;

    private int modoVista = 0; 
    private boolean detectarUsuario = true; 
    private int alphaBlue = 50;
    private int sMin = 30, vMin = 30, hMin = 0, hMax = 35; 
    private int gauss = 15, speckle = 0, kernel = 3;

    private Mat fondoMat;
    private static final String TAG = "MainActivity";
    private long lastFrameNs = 0;
    private double fps = 0.0;

    private native void procesarFrame(
            long rgbaAddr, long bgAddr, int modo, boolean detectarUsuario,
            int alphaBlue, int sMin, int vMin, int hMin, int hMax,
            int gauss, int speckle, int kernel
    );

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        cameraView = findViewById(R.id.cameraView);
        cameraView.setCameraIndex(0);
        statusText = findViewById(R.id.statusText);
        Spinner modoSpinner = findViewById(R.id.modoSpinner);
        CheckBox invertirCheck = findViewById(R.id.invertirCheck);

        bindViews();

        cameraView.setCvCameraViewListener(this);
        cameraView.setVisibility(SurfaceView.VISIBLE);

        String[] opcionesModo = {
            "Mosaico 3x3 (Entrega)", "Original (Cámara)", "Efecto Karl Struss",
            "Máscara (Blanco=Mantiene)", "Solo Usuario (Fondo Negro)",
            "Fondo Reemplazado", "Pipeline Completo"
        };
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, opcionesModo);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        modoSpinner.setAdapter(adapter);
        modoSpinner.setSelection(modoVista);

        modoSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(AdapterView<?> p, View v, int pos, long id) { modoVista = pos; }
            @Override public void onNothingSelected(AdapterView<?> p) {}
        });

        invertirCheck.setText("Rango = Usuario");
        invertirCheck.setChecked(detectarUsuario);
        invertirCheck.setOnCheckedChangeListener((b, isChecked) -> detectarUsuario = isChecked);

        setupSeekBars();

        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, 473);
        } else {
            habilitarCamara();
        }
    }

    private void bindViews() {
        alphaBlueText = findViewById(R.id.alphaBlueText);
        sMinText = findViewById(R.id.sMinText);
        vMinText = findViewById(R.id.vMinText);
        hMinText = findViewById(R.id.hMinText);
        hMaxText = findViewById(R.id.hMaxText);
        gaussText = findViewById(R.id.gaussText);
        speckleText = findViewById(R.id.speckleText);
        kernelText = findViewById(R.id.kernelText);
    }

    private void setupSeekBars() {
        configurarSeek(R.id.alphaBlueSeek, v -> { alphaBlue = v; alphaBlueText.setText(String.format(Locale.getDefault(), "Alpha Karl Struss: %.2f", v/100.0)); });
        configurarSeek(R.id.sMinSeek, v -> { sMin = v; sMinText.setText("Saturación Mínima: " + v); });
        configurarSeek(R.id.vMinSeek, v -> { vMin = v; vMinText.setText("Brillo Mínimo: " + v); });
        configurarSeek(R.id.hMinSeek, v -> { hMin = v; hMinText.setText("Hue Mínimo: " + v); });
        configurarSeek(R.id.hMaxSeek, v -> { hMax = v; hMaxText.setText("Hue Máximo: " + v); });
        configurarSeek(R.id.gaussSeek, v -> { gauss = v; gaussText.setText("Ruido Gaussiano: " + v); });
        configurarSeek(R.id.speckleSeek, v -> { speckle = v; speckleText.setText(String.format(Locale.getDefault(), "Ruido Speckle: %.2f", v/100.0)); });
        configurarSeek(R.id.kernelSeek, v -> { kernel = v * 2 + 1; kernelText.setText(String.format(Locale.getDefault(), "Kernel Mediana: %dx%d", kernel, kernel)); });
    }

    private void configurarSeek(int id, SeekBarCallback cb) {
        SeekBar s = findViewById(id);
        cb.onVal(s.getProgress());
        s.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar sb, int p, boolean f) { cb.onVal(p); }
            @Override public void onStartTrackingTouch(SeekBar sb) {}
            @Override public void onStopTrackingTouch(SeekBar sb) {}
        });
    }

    private interface SeekBarCallback { void onVal(int v); }

    private void habilitarCamara() {
        if (!OpenCVLoader.initLocal()) return;
        cameraView.setCameraPermissionGranted();
        cameraView.setMaxFrameSize(1280, 720);
        cameraView.enableView();
    }

    @Override public void onCameraViewStarted(int width, int height) {
        lastFrameNs = 0; 
        if (fondoMat == null) fondoMat = new Mat();
        
        try {
            // Buscamos la imagen fondo.jpg en res/drawable
            int resId = getResources().getIdentifier("verde", "drawable", getPackageName());
            if (resId != 0) {
                Bitmap bmp = BitmapFactory.decodeResource(getResources(), resId);
                if (bmp != null) {
                    Utils.bitmapToMat(bmp, fondoMat);
                    Imgproc.cvtColor(fondoMat, fondoMat, Imgproc.COLOR_RGBA2BGR);
                    Log.d(TAG, "FONDO CARGADO EXITOSAMENTE: " + fondoMat.cols() + "x" + fondoMat.rows());
                } else {
                    Log.e(TAG, "La imagen existe pero no se pudo decodificar (Bitmap is null)");
                }
            } else {
                Log.e(TAG, "NO SE ENCONTRÓ LA IMAGEN 'verde.jpg' EN res/drawable");
            }
        } catch (Exception e) { 
            Log.e(TAG, "Error crítico cargando fondo: " + e.getMessage()); 
        }
    }

    @Override public void onCameraViewStopped() { if (fondoMat != null) fondoMat.release(); }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat rgba = inputFrame.rgba();
        if (modoVista != 1) {
            procesarFrame(rgba.getNativeObjAddr(), fondoMat.getNativeObjAddr(), modoVista, detectarUsuario, alphaBlue, sMin, vMin, hMin, hMax, gauss, speckle, kernel);
        }
        long now = System.nanoTime();
        if (lastFrameNs != 0) { fps = 0.9 * fps + 0.1 * (1_000_000_000.0 / (now - lastFrameNs)); }
        lastFrameNs = now;
        runOnUiThread(() -> statusText.setText(String.format(Locale.getDefault(), "FPS=%.2f | Modo=%d | H=[%d,%d]", fps, modoVista, hMin, hMax)));
        return rgba;
    }
}
