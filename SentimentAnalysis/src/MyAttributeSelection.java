import weka.attributeSelection.*;
import weka.core.*;
import weka.core.converters.ArffSaver;
import weka.filters.*;
import weka.filters.unsupervised.attribute.Remove;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import java.util.StringTokenizer;

public class MyAttributeSelection {

	Instances data;
	Instances newData;

	MyAttributeSelection()
	{

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

		//		System.out.println(data.numClasses());
		//		System.out.println(data.numAttributes());
		//		
		//		System.out.println(data.instance(3).numValues());
		//		System.out.println(data.instance(3).isMissing(16));
		//		System.out.println(data.instance(3).value(data.numAttributes()-1) == 0.0);
	}

	/**
	 * uses the filter
	 */
	public void cfsGreedyUseFilter(boolean searchBackwards, boolean useStartSet, String outputFile) {
		weka.filters.supervised.attribute.AttributeSelection filter = new weka.filters.supervised.attribute.AttributeSelection();
		CfsSubsetEval eval = new CfsSubsetEval();
		GreedyStepwise search = new GreedyStepwise();
		search.setSearchBackwards(searchBackwards);

		try{
			if(useStartSet)
				search.setStartSet(this.getStartSet());
		}
		catch(Exception e)
		{
			System.out.println("Exception in useFilter greedy - setStartSet");
			e.printStackTrace();
		}

		filter.setEvaluator(eval);
		filter.setSearch(search);
		try {
			filter.setInputFormat(data);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			System.out.println("Exception in useFilter greedy - setInputFormat");
			e.printStackTrace();
		}
		try {
			newData = Filter.useFilter(data, filter);
			System.out.println(newData);

			//save in new arff file
			ArffSaver saver = new ArffSaver();
			saver.setInstances(newData);
			saver.setFile(new File("resources//"+outputFile+".arff"));
			//saver.setDestination(new File("resources//"+outputFile+".arff"));   // **not** necessary in 3.5.4 and later
			saver.writeBatch();

		} catch (Exception e) {
			// TODO Auto-generated catch block
			System.out.println("Exception in useFilter greedy - useFilter");
			e.printStackTrace();
		}

	}

	/**
	 * uses the low level approach
	 */
	public void cfsGreedyLowLevelSelection(boolean searchBackwards, boolean useStartSet)
	{
		AttributeSelection attsel = new AttributeSelection();  // package weka.attributeSelection!
		CfsSubsetEval eval = new CfsSubsetEval();
		GreedyStepwise search = new GreedyStepwise();
		search.setSearchBackwards(searchBackwards);
		try{
			if(useStartSet)
				search.setStartSet(this.getStartSet());
		}
		catch(Exception e)
		{
			System.out.println("Exception in lowLevelSelection greedy - setStartSet");
			e.printStackTrace();
		}
		attsel.setEvaluator(eval);
		attsel.setSearch(search);

		try {
			attsel.SelectAttributes(data);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			System.out.println("Exception in lowLevelSelection greedy - selectAttributes");
			e.printStackTrace();
		}
		// obtain the attribute indices that were selected
		try {
			System.out.println("Full result description:");
			System.out.println(attsel.toResultsString());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			System.out.println("Exception in lowLevelSelection greedy - selectedAttributes");
			e.printStackTrace();
		}

	}

	public void cfsBestFirstLowLevelSelection(boolean useStartSet)
	{
		AttributeSelection attsel = new AttributeSelection();  // package weka.attributeSelection!
		CfsSubsetEval eval = new CfsSubsetEval();
		BestFirst search = new BestFirst();
		try{
			if(useStartSet)
				search.setStartSet(this.getStartSet());
		}
		catch(Exception e)
		{
			System.out.println("Exception in lowLevelSelection bestFirst - setStartSet");
			e.printStackTrace();
		}
		attsel.setEvaluator(eval);
		attsel.setSearch(search);

		try {
			attsel.SelectAttributes(data);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			System.out.println("Exception in lowLevelSelection bestFirst - selectAttributes");
			e.printStackTrace();
		}
		// obtain the attribute indices that were selected
		try {
			System.out.println("Full result description:");
			System.out.println(attsel.toResultsString());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			System.out.println("Exception in lowLevelSelection bestFirst - selectedAttributes");
			e.printStackTrace();
		}
	}

	public void cfsBestFirstUseFilter(boolean useStartSet, String outputFile)
	{
		weka.filters.supervised.attribute.AttributeSelection filter = new weka.filters.supervised.attribute.AttributeSelection();
		CfsSubsetEval eval = new CfsSubsetEval();
		BestFirst search = new BestFirst();

		try{
			if(useStartSet)
				search.setStartSet(this.getStartSet());
		}
		catch(Exception e)
		{
			System.out.println("Exception in useFilter bestFirst - setStartSet");
			e.printStackTrace();
		}

		filter.setEvaluator(eval);
		filter.setSearch(search);
		try {
			filter.setInputFormat(data);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			System.out.println("Exception in useFilter bestFirst - setInputFormat");
			e.printStackTrace();
		}
		try {
			newData = Filter.useFilter(data, filter);
			System.out.println(newData);

			//save in new arff file
			ArffSaver saver = new ArffSaver();
			saver.setInstances(newData);
			saver.setFile(new File("resources//"+outputFile+".arff"));
			//saver.setDestination(new File("resources//"+outputFile+".arff"));   // **not** necessary in 3.5.4 and later
			saver.writeBatch();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			System.out.println("Exception in useFilter bestFirst - useFilter");
			e.printStackTrace();
		}
	}

	public void RankerCrossValidation(int folds, int seed, String name)
	{
		try {
			PrintWriter out = new PrintWriter(new FileWriter("resources//rankerCVoutputfile_"+name+".txt"));

			AttributeSelection attsel = new AttributeSelection();
			InfoGainAttributeEval eval = new InfoGainAttributeEval();
			Ranker search = new Ranker();

			attsel.setEvaluator(eval);
			attsel.setSearch(search);

			try {
				attsel.setFolds(folds);
				attsel.setSeed(seed);

				Instances randData = this.randomizeData(folds, seed);

				for (int n = 0; n < folds; n++)
				{
					System.out.println("Fold " + n);

					AttributeSelection foldAttsel = new AttributeSelection();
					InfoGainAttributeEval foldEval = new InfoGainAttributeEval();
					Ranker foldSearch = new Ranker();

					foldAttsel.setEvaluator(foldEval);
					foldAttsel.setSearch(foldSearch);

					Instances train = randData.trainCV(folds, n);
					Instances test = randData.testCV(folds, n);

					//foldAttsel.SelectAttributes(train);
					out.println("Result for fold "+n);
					//out.println(foldAttsel.toResultsString());

					attsel.selectAttributesCVSplit(train);
					out.println(attsel.CVResultsString());
					//				System.out.println("Number of attributes selected in fold " + n);
					//				System.out.println(attsel.numberAttributesSelected());
				}

				System.out.println(attsel.CVResultsString());
				out.println("Overall results");
				out.println(attsel.CVResultsString());
			} catch (Exception e) {
				// TODO Auto-generated catch block
				System.out.println("Exception at ranker cross validation");
				e.printStackTrace();
			}

			out.close();

		} catch (IOException e1) {
			// TODO Auto-generated catch block
			System.out.println("Exception printwriter in ranker cross validation");
			e1.printStackTrace();
		}
	}

	public void CfsCrossValidation(int folds, int seed, String name, String outputFile)
	{
		try {
			PrintWriter out = new PrintWriter(new FileWriter("resources//cfsCVoutputfile_"+name+".txt"));

			AttributeSelection attsel = new AttributeSelection();
			CfsSubsetEval eval = new CfsSubsetEval();
			BestFirst search = new BestFirst();

			attsel.setEvaluator(eval);
			attsel.setSearch(search);

			try {
				attsel.setFolds(folds);
				attsel.setSeed(seed);
				Instances randData = this.randomizeData(folds, seed);

				for (int n = 0; n < folds; n++)
				{
					System.out.println("Fold " + n);

					//AttributeSelection foldAttsel = new AttributeSelection();
					//CfsSubsetEval foldEval = new CfsSubsetEval();
					//BestFirst foldSearch = new BestFirst();

					//foldAttsel.setEvaluator(foldEval);
					//foldAttsel.setSearch(foldSearch);

					Instances train = randData.trainCV(folds, n);
					Instances test = randData.testCV(folds, n);

					//foldAttsel.SelectAttributes(train);
					out.println("Result for fold "+n);
					//out.println(foldAttsel.toResultsString());

					attsel.selectAttributesCVSplit(train);
					out.println(attsel.CVResultsString());
					//				System.out.println("Number of attributes selected in fold " + n);
					//				System.out.println(attsel.numberAttributesSelected());
				}

				System.out.println(attsel.CVResultsString());
				out.println("Overall results");
				out.println(attsel.CVResultsString());

				//reduce arff based on cross validation
				//System.out.println("Reduce dimensionality");
				//attsel.CrossValidateAttributes();

				/*newData = attsel.reduceDimensionality(data);

				//save in new arff file
				 ArffSaver saver = new ArffSaver();
				 saver.setInstances(newData);
				 saver.setFile(new File("resources//"+outputFile+".arff"));				
				 saver.writeBatch();*/

			} catch (Exception e) {
				// TODO Auto-generated catch block
				System.out.println("Exception at cfs cross validation");
				e.printStackTrace();
			}

			out.close();

		} catch (IOException e1) {
			// TODO Auto-generated catch block
			System.out.println("Exception printwriter in cfs cross validation");
			e1.printStackTrace();
		}
	}

	public Instances getNewInstances()
	{
		return newData;
	}

	private String getStartSet()
	{
		String indices = "";
		AttributeSelection attsel = new AttributeSelection();
		InfoGainAttributeEval eval = new InfoGainAttributeEval();
		Ranker search = new Ranker();

		attsel.setEvaluator(eval);
		attsel.setSearch(search);

		try {
			attsel.SelectAttributes(data);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			System.out.println("Exception in computing startSet");
			e.printStackTrace();
		}

		try {
			double[][] ranked = attsel.rankedAttributes();
			System.out.println("Computed start set !");
			int i = 0;
			boolean first = true;
			while(true)
			{
				if(ranked[i][1] == 0)
					break;
				else
				{
					if(first)
					{
						indices += ((int)ranked[i][0]+1);
						first = false;
					}
					else
						indices += "," + ((int)ranked[i][0]+1);

					i++;
				}
			}
			System.out.println(indices);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			System.out.println("Exception in getStartSet - selectedAttributes");
			e.printStackTrace();
		}

		return indices;
	}

	private Instances randomizeData(int folds, int seed)
	{
		Random rand = new Random(seed);   // create seeded number generator
		Instances randData = new Instances(data);   // create copy of original data
		randData.randomize(rand);
		if (randData.classAttribute().isNominal())
			randData.stratify(folds);

		return randData;

	}

	public HashMap<Integer,Boolean> readCSVFile(String csvName)
	{
		String csvFile = "resources//"+csvName+".csv";

		HashMap<Integer,Boolean> index = new HashMap<Integer,Boolean>();

		try{
			//create BufferedReader to read csv file
			BufferedReader br = new BufferedReader(new FileReader(csvFile));
			String line = "";
			StringTokenizer st = null;

			int lineNumber = 0; 
			int tokenNumber = 0;
			int auxNumber1, auxNumber2;
			boolean keepFeature = false;

			//read semicolon separated file line by line
			while ((line = br.readLine()) != null) {
				lineNumber++;

				//use comma as token separator
				st = new StringTokenizer(line, ";");

				while (st.hasMoreTokens()) {
					tokenNumber++;

					String token = st.nextToken();

					if(tokenNumber == 1)
					{
						auxNumber1 = Integer.parseInt(token);
						if(auxNumber1 > 0)
							keepFeature = true;
					}

					if(tokenNumber == 3)
					{
						if(keepFeature)
						{
							auxNumber2 = Integer.parseInt(token);
							index.put(auxNumber2, true);
							System.out.println("Am selectat feature cu index " + auxNumber2);
						}
					}
				}

				//reset token number
				tokenNumber = 0;
				keepFeature = false;
			}

			return index;

		} catch (Exception e) {
			System.err.println("CSV file cannot be read : " + e);
			return null;
		}
	}

	public void featureSelectedData(HashMap<Integer,Boolean> index, String outputFile)
	{
		Remove remove = new Remove();

		Integer[] indicesAux = index.keySet().toArray(new Integer[index.keySet().size()]);

		int [] indices = new int[indicesAux.length+1];
		for(int i = 0; i<indicesAux.length; i++)
			indices[i] = indicesAux[i];
		indices[indicesAux.length] = data.numAttributes() - 1;

		remove.setAttributeIndicesArray(indices);
		remove.setInvertSelection(true);
		try {
			remove.setInputFormat(data);
			newData =  Filter.useFilter(data, remove);

			//			int j = 0;
			//			while(j<newData.numInstances())
			//			{
			//				if((newData.instance(j).numValues() == 0) || ((newData.instance(j).numValues() == 1) && (newData.instance(j).value(newData.numAttributes()-1) != 0.0) ) )
			//					newData.delete(j);
			//				else
			//					j++;
			//			}

			System.out.println("Nr. de features pt newData = " + newData.numAttributes());

		} catch (Exception e1) {
			// TODO Auto-generated catch block
			System.out.println("Exception in featureSelectedData - filter");
			e1.printStackTrace();
		}

		//save in new arff file
		ArffSaver saver = new ArffSaver();
		saver.setInstances(newData);
		try {
			saver.setFile(new File("resources//"+outputFile+".arff"));
			saver.writeBatch();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println("Exception in featureSelectedData - write new arff");
			e.printStackTrace();
		}				

	}

	public void manualAttributeSelection(String outputFile)
	{
		Remove remove = new Remove();
		remove.setInvertSelection(true);

		ArrayList<Integer> selectedAttributes = new ArrayList<Integer>();

		int numberOfAttributes = data.numAttributes() - 1;
		for(int i = 0; i < numberOfAttributes; i++)
		{
			int numberOfInstances = data.numInstances();
			int countPosEx = 0;
			int countNegEx = 0;
			for(int j = 0; j < numberOfInstances; j++)
			{
				if(data.instance(j).value(i) != 0.0)
					if(data.instance(j).value(numberOfAttributes) == 0.0)
						countNegEx++;
					else
						countPosEx++;				
			}

			System.out.println(data.attribute(i)+" "+countNegEx+" "+countPosEx);
			if((countNegEx+countPosEx) > (numberOfInstances / 200)) //hardcoded value
			{
				if(countNegEx == 0 || countPosEx == 0)
					selectedAttributes.add(i);
				else
					if((countPosEx/countNegEx > 1.5) ||  (countNegEx/countPosEx > 1.5))
						selectedAttributes.add(i);
			}
		}

		System.out.println("Intial!!!  "+data.numAttributes()+" "+data.numInstances());
		System.out.println("Number of selected attributes = " + selectedAttributes.size());

		Integer[] indicesAux = selectedAttributes.toArray(new Integer[selectedAttributes.size()]);
		int [] indices = new int[indicesAux.length+1];
		for(int i = 0; i<indicesAux.length; i++)
			indices[i] = indicesAux[i];
		indices[indicesAux.length] = data.numAttributes() - 1;

		remove.setAttributeIndicesArray(indices);

		try {
			remove.setInputFormat(data);
			newData =  Filter.useFilter(data, remove);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			System.out.println("Exception in manual feature selection - filter");
			e1.printStackTrace();
		}

		//save in new arff file
		ArffSaver saver = new ArffSaver();
		saver.setInstances(newData);
		try {
			saver.setFile(new File("resources//"+outputFile+".arff"));
			saver.writeBatch();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println("Exception in featureSelectedData - write new arff");
			e.printStackTrace();
		}	
	}
	
	public void removeNeutralInstances(String outputFile)
	{	
		int numberOfAttributes = data.numAttributes();

		int j = 0;
		while(j < data.numInstances())
		{
			//System.out.println(data.instance(j).value(numberOfAttributes - 1));
			if(data.instance(j).value(numberOfAttributes - 1) == 1.0)
				data.delete(j);
			else
				j++;
		}
		
		//save in new arff file
		ArffSaver saver = new ArffSaver();
		saver.setInstances(data);
		try {
			saver.setFile(new File("resources//"+outputFile+".arff"));
			saver.writeBatch();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println("Exception in featureSelectedData - write new arff");
			e.printStackTrace();
		}
		
	}
	
	public void removeLastNAttributes(int n, String outputFile)
	{
		Remove remove = new Remove();
		
		int [] indices = new int[n];
		for(int i=1; i<=n; i++)
			indices[i-1] = (data.numAttributes() - 1) - i;
		
		remove.setAttributeIndicesArray(indices);
		
		try {
			remove.setInputFormat(data);
			newData =  Filter.useFilter(data, remove);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			System.out.println("Exception in manual feature selection - filter");
			e1.printStackTrace();
		}

		//save in new arff file
		ArffSaver saver = new ArffSaver();
		saver.setInstances(newData);
		try {
			saver.setFile(new File("resources//"+outputFile+".arff"));
			saver.writeBatch();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println("Exception in featureSelectedData - write new arff");
			e.printStackTrace();
		}	
	}

}


