package com.rafasoft.gallerai.tagger;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

/** 
 * The model
 * 
 * @author Rafa
 *
 */
public class Model{
	public byte[] graphDef;
	public List<String> labels;
	
	public Model(){
		try {
			graphDef = Files.readAllBytes(Paths.get("src", "main", "resources", "models", "tensorflow_inception_graph.pb"));
			labels = Files.readAllLines(Paths.get("src", "main", "resources", "models", "imagenet_comp_graph_label_strings.txt"), Charset.forName("UTF-8"));	
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
	}		
}