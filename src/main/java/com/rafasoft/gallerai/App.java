package com.rafasoft.gallerai;

import java.io.IOException;

import com.rafasoft.gallerai.tagger.ImageTagger;

/**
 * The App!
 *
 */
public class App {
	private static String imagesPath = "src\\main\\resources\\images";

	public static void main(String[] args) {
		String current = null;
		try {
			current = new java.io.File(".").getCanonicalPath();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println("Current dir:" + current);
		ImageTagger tagger = new ImageTagger();
		try {
			tagger.tagImages(imagesPath);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
