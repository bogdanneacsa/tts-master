import java.io.IOException;


public class Main {
	
	public static void BayesMultinomial(String arff, boolean runFeatureSelection)
	{
		MyAttributeSelection atSel = new MyAttributeSelection();
		try {
			atSel.loadFile("resources//"+arff+".arff");
		} catch (IOException e1) {
			System.out.println("Exception - in main selection load file");
			e1.printStackTrace();
		}
		
		Classifier classifier = new Classifier(new weka.classifiers.bayes.NaiveBayesMultinomialText());
		
		try {
			classifier.loadFile("resources//"+arff+".arff");
			classifier.crossValidation(10, "naiveBayesM_"+arff);
			
			/*if(runFeatureSelection)
			{
				atSel.loadFile("resources//"+arff+".arff");
				atSel.cfsBestFirstUseFilter(false, arff+"_filtered");
				classifier.setData(atSel.getNewInstances());
			}
			else
				classifier.loadFile("resources//"+arff+"_filtered.arff");
			classifier.crossValidation(10, "naiveBayesM_"+arff+"_featureSel");
			
			if(runFeatureSelection)
			{
				atSel.loadFile("resources//"+arff+".arff");
				atSel.CfsCrossValidation(10, 1, arff, arff+"_CVfiltered");
				classifier.setData(atSel.getNewInstances());
			}
			else
				classifier.loadFile("resources//"+arff+"_CVfiltered.arff");
			classifier.crossValidation(10, "naiveBayesM_"+arff+"_CVfeatureSel");*/
			
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println("Exception - in main Test first classifier");
			e.printStackTrace();
		}
	}
	
	public static void SMOPolyKernel(String arff, boolean runFeatureSelection)
	{
		MyAttributeSelection atSel = new MyAttributeSelection();
		try {
			atSel.loadFile("resources//"+arff+".arff");
		} catch (IOException e1) {
			System.out.println("Exception - in main selection load file");
			e1.printStackTrace();
		}
		
		weka.classifiers.functions.SMO SMOclassifier = new weka.classifiers.functions.SMO();
		SMOclassifier.setC(1.0);
		weka.classifiers.functions.supportVector.PolyKernel kernel = new weka.classifiers.functions.supportVector.PolyKernel();
		kernel.setExponent(1.0);
		SMOclassifier.setKernel(kernel);
		
		Classifier classifier = new Classifier(SMOclassifier);
		
		try {
			classifier.loadFile("resources//"+arff+".arff");
			classifier.crossValidation(10, "polyKernel_"+arff);
			
			/*if(runFeatureSelection)
			{
				atSel.loadFile("resources//"+arff+".arff");
				atSel.cfsBestFirstUseFilter(false, arff+"_filtered");
				classifier.setData(atSel.getNewInstances());
			}
			else
				classifier.loadFile("resources//"+arff+"_filtered.arff");
			classifier.crossValidation(10, "polyKernel_"+arff+"_featureSel");
			
			if(runFeatureSelection)
			{
				atSel.loadFile("resources//"+arff+".arff");
				atSel.CfsCrossValidation(10, 1, arff, arff+"_CVfiltered");
				classifier.setData(atSel.getNewInstances());
			}
			else
				classifier.loadFile("resources//"+arff+"_CVfiltered.arff");
			classifier.crossValidation(10, "polyKernel_"+arff+"_CVfeatureSel");*/
			
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println("Exception - in main Test first classifier");
			e.printStackTrace();
		}
	}
	
	public static void SMORBFKernel(String arff,  boolean runFeatureSelection)
	{
		MyAttributeSelection atSel = new MyAttributeSelection();
		try {
			atSel.loadFile("resources//"+arff+".arff");
		} catch (IOException e1) {
			System.out.println("Exception - in main selection load file");
			e1.printStackTrace();
		}
		
		weka.classifiers.functions.SMO SMOclassifier = new weka.classifiers.functions.SMO();
		SMOclassifier.setC(1.0);
		weka.classifiers.functions.supportVector.RBFKernel kernel = new weka.classifiers.functions.supportVector.RBFKernel();
		kernel.setGamma(0.01);
		SMOclassifier.setKernel(kernel);
		
		Classifier classifier = new Classifier(SMOclassifier);
		
		try {
			classifier.loadFile("resources//"+arff+".arff");
			classifier.crossValidation(10, "rbfKernel_"+arff);
			
			/*if(runFeatureSelection)
			{
				atSel.loadFile("resources//"+arff+".arff");
				atSel.cfsBestFirstUseFilter(false, arff+"_filtered");
				classifier.setData(atSel.getNewInstances());
			}
			else
				classifier.loadFile("resources//"+arff+"_filtered.arff");
			classifier.crossValidation(10, "rbfKernel_"+arff+"_featureSel");
			
			if(runFeatureSelection)
			{
				atSel.loadFile("resources//"+arff+".arff");
				atSel.CfsCrossValidation(10, 1, arff, arff+"_CVfiltered");
				classifier.setData(atSel.getNewInstances());
			}
			else
				classifier.loadFile("resources//"+arff+"_CVfiltered.arff");
			classifier.crossValidation(10, "rbfKernel_"+arff+"_CVfeatureSel");*/
			
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println("Exception - in main Test first classifier");
			e.printStackTrace();
		}
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
//		BayesMultinomial("isear_combined_tfidf", true);
//		SMOPolyKernel("isear_combined_tfidf", false);
//		SMORBFKernel("isear_combined_tfidf", false);
//		
//		BayesMultinomial("isear_combined_binary", true);
//		SMOPolyKernel("isear_combined_binary", false);
//		SMORBFKernel("isear_combined_binary", false);
//		
//		BayesMultinomial("isear_turney_tfidf", true);
//		SMOPolyKernel("isear_turney_tfidf", false);
//		SMORBFKernel("isear_turney_tfidf", false);
//		
//		BayesMultinomial("isear_turney_binary", true);
//		SMOPolyKernel("isear_turney_binary", false);
//		SMORBFKernel("isear_turney_binary", false);
		
		//MyPOSTagger tagger = new MyPOSTagger("models//english-bidirectional-distsim.tagger");
		//tagger.taggFile("resources//sample-input.txt");
		
		MyAttributeSelection atSel1 = new MyAttributeSelection();
		MyAttributeSelection atSel2 = new MyAttributeSelection();
		MyAttributeSelection atSel3 = new MyAttributeSelection();
		MyAttributeSelection atSel4 = new MyAttributeSelection();
		try {
			atSel1.loadFile("resources//isear_combined_tfidf.arff");
			atSel2.loadFile("resources//isear_combined_binary.arff");
			atSel3.loadFile("resources//isear_turney_tfidf.arff");
			atSel4.loadFile("resources//isear_turney_binary.arff");
//			//atSel1.featureSelectedData(atSel1.readCSVFile("CFS_fairytale_tfidf"), "fairytale_balance_tfidf_cfs2");
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			System.out.println("Exception - in main selection load file");
			e1.printStackTrace();
		}
//		
//		atSel.RankerCrossValidation(10, 1, "fairytale_balance_tfidf");
//		atSel.RankerCrossValidation(10, 2);
//		atSel.RankerCrossValidation(10, 5);
//		
		atSel1.CfsCrossValidation(10, 1, "isear_combined_tfidf", "");
		atSel2.CfsCrossValidation(10, 1, "isear_combined_binary", "");
		atSel3.CfsCrossValidation(10, 1, "isear_turney_tfidf", "");
		atSel4.CfsCrossValidation(10, 1, "isear_turney_binary", "");
//		atSel.CfsCrossValidation(10, 2);
//		atSel.CfsCrossValidation(10, 5);
		
//		long startGreedyBack = System.currentTimeMillis();
//		System.out.println("cfs + greedy Low level feature selection - backwards search, without startSet");
//		atSel.cfsGreedyLowLevelSelection(true, false);
//		long endGreedyBack = System.currentTimeMillis();
//		System.out.println("Execution time  was "+(endGreedyBack-startGreedyBack)/1000+" s.");

		
//		System.out.println("cfs + greedy Low level feature selection - without startSet");
//		atSel.cfsGreedyLowLevelSelection(false, false);
//		
//		System.out.println("cfs + greedy Low level feature selection - with startSet");
//		atSel.cfsGreedyLowLevelSelection(false, true);
		
//		System.out.println("cfs + bestFirst Low level feature selection - without startSet");
//		atSel.cfsBestFirstLowLevelSelection(false);
//		
//		System.out.println("cfs + bestFirst Low level feature selection - with startSet");
//		atSel.cfsBestFirstLowLevelSelection(true);
		
//		System.out.println("Filter data");
//		atSel.cfsBestFirstUseFilter(false, "fairytale_balance_tfidf_filtered");
//		
//		System.out.println("Test first classifier");
//		long startFirstclass = System.currentTimeMillis();
		
//		Classifier classifier = new Classifier(new weka.classifiers.bayes.NaiveBayesMultinomialText());
//		
//		try {
//			classifier.loadFile("resources//fairytale_balance_tfidf_filtered.arff");
//			classifier.crossValidation(10, "naiveBayesM_fairytale_balance_tfidf_init");
//			//atSel.cfsGreedyUseFilter(false, true);
//			
//			atSel.loadFile("resources//fairytale_balance_tfidf.arff");
//			atSel.cfsBestFirstUseFilter(false);
//			classifier.setData(atSel.getNewInstances());
//			classifier.crossValidation(10, "naiveBayesM_fairytale_balance_tfidf_featureSel");
//			
//			atSel.loadFile("resources//fairytale_balance_tfidf.arff");
//			atSel.CfsCrossValidation(10, 1, "fairytale_balance_tfidf");
//			classifier.setData(atSel.getNewInstances());
//			classifier.crossValidation(10, "naiveBayesM_fairytale_balance_tfidf_CVfeatureSel");
//			
//			
//		} catch (IOException e) {
//			// TODO Auto-generated catch block
//			System.out.println("Exception - in main Test first classifier");
//			e.printStackTrace();
//		}
		
//		classifier.crossValidation(10, "binary_balance_fairytale");

		/*long endFirstClass = System.currentTimeMillis();
		
		System.out.println("Execution time for first classifier was "+(endFirstClass-startFirstclass)/1000+" s.");
		
		System.out.println("Test second classifier");
		
		classifier.setClassifier(new weka.classifiers.functions.SMO());
		
		long startSecondClass = System.currentTimeMillis();
		
		try {
			classifier.loadFile("resources//fairytales.arff");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println("Exception - in main Test second classifier");
			e.printStackTrace();
		}
		
		classifier.crossValidation(10);
		
		long endSecondClassifier = System.currentTimeMillis();
		System.out.println("Execution time for second classifier was "+(endSecondClassifier-startSecondClass)/1000+" s.");
		*/
	}

}
