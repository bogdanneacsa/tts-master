import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.evaluation.Evaluation;

public class Classifier {

	Instances data;
	Instances testData; 
	private static final String classifierStorage = "resources//classfierObject";
	/*
	 * Can change the type of the classifier to any of the weka classifiers that can
	 * handle both binary and real-values features.
	 */
	weka.classifiers.AbstractClassifier classifier;
	//ClassificationViaRegression classifier = new ClassificationViaRegression();

	public Classifier(weka.classifiers.AbstractClassifier classifier)
	{
		this.classifier = classifier;
	}
	
	public void setClassifier(weka.classifiers.AbstractClassifier classifier)
	{
		this.classifier = classifier;
	}

	public double classify(Instance key) throws Exception
	{
		classifier.distributionForInstance(key);
		return classifier.classifyInstance(key);
	}


	/*
	 * Builds the classifier form a default placed file containing the training instances.
	 */
	public void buildClassifier() throws Exception
	{
			String filename = "resources\\trainingData.arff";
			loadFile(filename);
			classifier.buildClassifier(data);
			this.saveToFile();	
	}

	/*
	 * Saves the classifier into a file in a default location.
	 */
	public void saveToFile()
	{
		try {
			FileOutputStream fos = new FileOutputStream(classifierStorage);
			ObjectOutputStream oos = new ObjectOutputStream(fos);

			oos.writeObject(this.classifier);
			oos.close();
		} catch (FileNotFoundException e) {
			System.out.println("File not found exception - save classifier!");
		} catch (IOException ex) {
			System.out.println("IO exception - save classifier!");
			ex.printStackTrace();
		}
	}	

	/*
	 * Loads data from a file.
	 */
	public void loadFile(String file) throws IOException
	{
		BufferedReader reader = new BufferedReader(new FileReader(file));
		data = new Instances(reader);
		reader.close();
		// setting class attribute
		data.setClassIndex(data.numAttributes() - 1);	
	}

	public void crossValidation(int noOfFolds, String name)
	{
		Evaluation eval;
				
		try {
			PrintWriter out = new PrintWriter(new FileWriter("resources//classCV_"+name+".txt"));
			
			eval = new Evaluation(data);
			eval.crossValidateModel(classifier, data, noOfFolds, new Random(1));
			System.out.println(eval.toSummaryString("\nResults\n======\n", true));
			out.println(eval.toSummaryString("\nResults\n======\n", true));
			System.out.println(eval.toClassDetailsString());
			out.println(eval.toClassDetailsString());
			
			System.out.println(eval.toMatrixString());
			out.println(eval.toMatrixString());
			
			out.close();
		}
		catch (Exception e) {
			// TODO Auto-generated catch block
			System.out.println("Exception - crossValidation!");
			e.printStackTrace();
		}
	}

	public void testClassifier()
	{
		Evaluation eval;
		try {
			eval = new Evaluation(data);
			eval.evaluateModel(classifier, testData);
			System.out.println(eval.toSummaryString("\nResults\n======\n", true));
			System.out.println(eval.toClassDetailsString());
		}
		catch (Exception e) {
			// TODO Auto-generated catch block
			System.out.println("Exception - testing classifier!");
			e.printStackTrace();
		}
	}

	/*
	 * Loads test data from a file.
	 */
	public void loadTestData(String file) throws IOException
	{		
		BufferedReader reader = new BufferedReader(new FileReader(file));
		testData = new Instances(reader);
		reader.close();
		// setting class attribute
		testData.setClassIndex(testData.numAttributes() - 1);
	}

	public Instances getData()
	{
		return data;
	}

	public void setData(Instances data)
	{
		this.data = data;
		data.setClassIndex(data.numAttributes() - 1);	
	}
	
}
