package com.rafasoft.gallerai.tagger;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

public class ImageCropper {
    private static final int WIDTH = 224;
    private static final int HEIGHT = 224;
    private static final int VERTICAL_SAMPLES = 10;
    private static final int HORIZONTAL_SAMPLES = 10;

    /**
     * Crops the passed image and returns a list of sub images including the
     * original one
     * 
     * @param image
     * @return
     */
    public static List<BufferedImage> getImages(BufferedImage image) {
        List<BufferedImage> images = new ArrayList<BufferedImage>();
        int maxWidth = image.getWidth();
        int maxHeight = image.getHeight();
        int horizontalInc = Math.max((maxWidth - WIDTH) / HORIZONTAL_SAMPLES, 0);
        int verticalInc = Math.max((maxHeight - HEIGHT) / VERTICAL_SAMPLES, 0);

        // System.out.println("Image (" + image.getWidth() + ", " +
        // image.getHeight() + ")");
        images.add(image);
        for (int h = 0; h < HORIZONTAL_SAMPLES; ++h) {
            for (int v = 0; v < VERTICAL_SAMPLES; ++v) {
                int width = Math.min(WIDTH, maxWidth - (h * horizontalInc));
                int height = Math.min(HEIGHT, maxHeight - (h * verticalInc));
                // System.out.println("Cropping (" + h * horizontalInc + ", " +
                // v * verticalInc + "), to ("
                // + (h * horizontalInc + width) + ", " + (v * verticalInc +
                // height) + ")");
                images.add(image.getSubimage(h * horizontalInc, v * verticalInc, width, height));
            }
        }
        // System.out.println("Done cropping");

        return images;
    }

}
