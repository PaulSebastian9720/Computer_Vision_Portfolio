package com.example.shapesignature;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    static {
        System.loadLibrary("shapesignature");
    }

    private static final String TAG = "ShapeSignature";
    private static final int PERM_CODE = 100;

    private static final java.util.Map<Integer, String> SPECIES = new java.util.HashMap<>();
    static {
        SPECIES.put(24, "Pittosporum tobira");
        SPECIES.put(25, "Nelumbo nucifera");
        SPECIES.put(26, "Acer palmatum");
        SPECIES.put(27, "Diospyros kaki");
        SPECIES.put(28, "Populus tomentosa");
        SPECIES.put(29, "Armeniaca mume");
        SPECIES.put(30, "Cinnamomum japonicum");
    }

    private PreviewView previewView;
    private TextView tvClase;
    private TextView tvDescriptores;
    private Button btnClasificar;

    private ExecutorService cameraExecutor;
    private volatile boolean classifyNext = false;

    private native String processFrame(long rgbaAddr);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        previewView    = findViewById(R.id.cameraView);
        tvClase        = findViewById(R.id.tvClase);
        tvDescriptores = findViewById(R.id.tvDescriptores);
        btnClasificar  = findViewById(R.id.btnClasificar);

        cameraExecutor = Executors.newSingleThreadExecutor();

        if (!OpenCVLoader.initLocal()) {
            Log.e(TAG, "Error al cargar OpenCV");
            tvClase.setText("Error: OpenCV no disponible");
            return;
        }
        Log.d(TAG, "OpenCV cargado: " + OpenCVLoader.OPENCV_VERSION);

        btnClasificar.setOnClickListener(v -> {
            classifyNext = true;
            tvClase.setText("Analizando...");
        });

        if (checkSelfPermission(Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, PERM_CODE);
        } else {
            startCamera();
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> future =
                ProcessCameraProvider.getInstance(this);

        future.addListener(() -> {
            try {
                ProcessCameraProvider provider = future.get();

                // Preview en tiempo real
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                // Análisis de frames para clasificación
                ImageAnalysis analysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                analysis.setAnalyzer(cameraExecutor, this::analyzeFrame);

                provider.unbindAll();
                provider.bindToLifecycle(this,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        analysis);

                Log.d(TAG, "CameraX iniciado");
                runOnUiThread(() -> tvClase.setText("Cámara lista — presiona Clasificar"));

            } catch (Exception e) {
                Log.e(TAG, "Error iniciando cámara: " + e.getMessage());
                runOnUiThread(() -> tvClase.setText("Error cámara: " + e.getMessage()));
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void analyzeFrame(ImageProxy imageProxy) {
        if (!classifyNext) {
            imageProxy.close();
            return;
        }
        classifyNext = false;

        // Convertir ImageProxy YUV a Mat RGBA via Bitmap
        try {
            Bitmap bitmap = imageProxy.toBitmap();
            Mat rgba = new Mat(bitmap.getHeight(), bitmap.getWidth(), CvType.CV_8UC4);
            Utils.bitmapToMat(bitmap, rgba);
            bitmap.recycle();

            // Rotar frame si es necesario (CameraX puede dar landscape en portrait)
            int rotation = imageProxy.getImageInfo().getRotationDegrees();
            if (rotation == 90) {
                org.opencv.core.Core.rotate(rgba, rgba, org.opencv.core.Core.ROTATE_90_CLOCKWISE);
            } else if (rotation == 270) {
                org.opencv.core.Core.rotate(rgba, rgba, org.opencv.core.Core.ROTATE_90_COUNTERCLOCKWISE);
            } else if (rotation == 180) {
                org.opencv.core.Core.rotate(rgba, rgba, org.opencv.core.Core.ROTATE_180);
            }

            String result = processFrame(rgba.getNativeObjAddr());
            rgba.release();
            runOnUiThread(() -> updateUI(result));

        } catch (Exception e) {
            Log.e(TAG, "Error analizando frame: " + e.getMessage());
            runOnUiThread(() -> tvClase.setText("Error: " + e.getMessage()));
        } finally {
            imageProxy.close();
        }
    }

    private void updateUI(String result) {
        if (result == null || result.isEmpty()) {
            tvClase.setText("ESPECIE CLASIFICADA: —");
            tvDescriptores.setText("Descriptor de Fourier: —");
            return;
        }

        String[] parts = result.split(";");
        int label = Integer.parseInt(parts[0]);

        if (label < 0) {
            tvClase.setText("ESPECIE CLASIFICADA: no detectada");
            tvDescriptores.setText("Sin forma válida en el encuadre");
            return;
        }

        String name = SPECIES.getOrDefault(label, "Clase " + label);
        tvClase.setText("Clase " + label + " — " + name);

        if (parts.length > 1) {
            StringBuilder sb = new StringBuilder("Descriptor de Fourier:\n");
            for (int i = 1; i < parts.length; i++) {
                try {
                    sb.append(String.format("F%d=%.4f", i, Float.parseFloat(parts[i])));
                    if (i < parts.length - 1) sb.append("  ");
                    if (i % 4 == 0) sb.append("\n");
                } catch (NumberFormatException ignored) {}
            }
            tvDescriptores.setText(sb.toString());
        }
    }

    @Override
    public void onRequestPermissionsResult(int code, String[] perms, int[] results) {
        super.onRequestPermissionsResult(code, perms, results);
        if (code == PERM_CODE && results.length > 0
                && results[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            tvClase.setText("Permiso de cámara denegado");
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraExecutor != null) cameraExecutor.shutdown();
    }
}
