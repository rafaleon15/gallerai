package com.rafasoft.gallerai.tagger;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import javax.imageio.ImageIO;

/**
 * Search for the images in the path and tags them
 * 
 * @author Rafa
 *
 */
public class ImageTagger {
    private ImageTagModel model;
    private static Float QUALITY_THRESHOLD = 0.5f;

    public ImageTagger() {
        ImageIO.setUseCache(false);
        model = new ImageTagModel();
    }

    public void tagImages(String path) throws IOException {
        File[] files = new File(path).listFiles();
        iterateAndTagImages(files);
    }

    /**
     * Iterate images recursively
     * 
     * @param files
     * @throws IOException
     */
    private void iterateAndTagImages(File[] files) throws IOException {
        for (File file : files) {
            if (file.isDirectory()) {
                System.out.println("Directory: " + file.getName());
                iterateAndTagImages(file.listFiles()); // Calls same method
                                                       // again.
            } else {
                System.out.println("File: " + file.getName());
                tagImage(file);
            }
        }
    }

    /**
     * Get the bytes from the image
     * 
     * @param image
     * @return
     * @throws IOException
     */
    private byte[] getBytes(BufferedImage image) throws IOException {
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        ImageIO.write(image, "jpg", outputStream);
        return outputStream.toByteArray();
    }

    /**
     * Crops and tags subimages
     * 
     * @param file
     * @throws IOException
     */
    private void tagImage(File file) throws IOException {
        BufferedImage image = ImageIO.read(file);
        List<BufferedImage> images = ImageCropper.getImages(image);
        HashMap<String, Float> tags = new HashMap<String, Float>();
        // Tag this image and its sub images
        for (BufferedImage im : images) {
            Set<ProbableTag> partialTags = model.tagImage(getBytes(im), QUALITY_THRESHOLD);
            Iterator<ProbableTag> i = partialTags.iterator();
            while (i.hasNext()) {
                ProbableTag pt = i.next();
                if (tags.containsKey(pt.tag)) {
                    tags.put(pt.tag, Math.max(pt.probability, tags.get(pt.tag)));
                } else {
                    tags.put(pt.tag, pt.probability);
                }

            }
        }
        System.out.println("Image: " + file.getName());
        System.out.println("Best matches:");
        Iterator<String> i = tags.keySet().iterator();
        while (i.hasNext()) {
            String tag = i.next();
            System.out.println("Label: " + tag + ", prob: " + tags.get(tag));
        }
        System.out.println("----------------");
    }

}
