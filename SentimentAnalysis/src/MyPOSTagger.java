import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.Sentence;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;


public class MyPOSTagger {

	private String model;
	
	MyPOSTagger(String model)
	{
		this.model = model;
	}

	public void taggFile(String file)
	{
		MaxentTagger tagger = new MaxentTagger(model);
		List<List<HasWord>> sentences;
		try {
			PrintWriter out = new PrintWriter(new FileWriter("resources//taggingOutput.txt"));
			sentences = MaxentTagger.tokenizeText(new BufferedReader(new FileReader(file)));
			for (List<HasWord> sentence : sentences) {
				ArrayList<TaggedWord> tSentence = tagger.tagSentence(sentence);
				out.println(Sentence.listToString(tSentence, false));
			}
			
			out.close();
		}
		catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			System.out.println("FileNotFoundException in tagging!");
			e.printStackTrace();
		}
		catch(IOException e1)
		{
			System.out.println("IOException in tagging!");
			e1.printStackTrace();
		}
	}

}
