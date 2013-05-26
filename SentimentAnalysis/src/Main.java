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
		
		Classifier classifier = new Classifier(new weka.classifiers.bayes.NaiveBayesMultinomial());
		
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
		
//		CombineInstances c = new CombineInstances("resources\\ise_processed");
//		c.combineInstance(3, "ise_processed");
//		c.combineInstance(5, "ise_processed");
//		c.combineInstance(7, "ise_processed");
		
//		BayesMultinomial("isear_one_plus_turney_tfidf3", true);
//		BayesMultinomial("isear_one_plus_turney_tfidf5", true);
//		BayesMultinomial("isear_one_plus_turney_tfidf7", true);
		
//		SMOPolyKernel("movies_1w_tfidf_fsManual", false);
//		SMORBFKernel("movies_1w_tfidf_fsManual", false);
		
//		BayesMultinomial("fairytale_balance_tfidf_ok_noNeutrals", true);
//		SMOPolyKernel("movies_1w_binary_fsManual", false);
//		SMORBFKernel("movies_1w_binary_fsManual", false);
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
		
//		MyAttributeSelection atSel1 = new MyAttributeSelection();
//		MyAttributeSelection atSel2 = new MyAttributeSelection();
//		MyAttributeSelection atSel3 = new MyAttributeSelection();
//		MyAttributeSelection atSel4 = new MyAttributeSelection();
//		try {
//			atSel1.loadFile("resources//isear_one_plus_turney_tfidf7.arff");
//			atSel2.loadFile("resources//isear_turney_tfidf_test.arff");
//			atSel3.loadFile("resources//isear_turney_tfidf.arff");
//			atSel4.loadFile("resources//isear_turney_binary.arff");
//			//atSel1.featureSelectedData(atSel1.readCSVFile("CFS_fairytale_tfidf"), "fairytale_balance_tfidf_cfs2");
//			atSel1.manualAttributeSelectionMulticlass("isear_one_plus_turney_tfidf7_fsManualMulti", 7);
//			atSel1.removeLastNAttributes(5, "fairytale_balance_tfidf_ok");
//			atSel1.loadFile("resources//fairytale_balance_tfidf_ok.arff");
//			atSel1.removeNeutralInstances("fairytale_balance_tfidf_ok_noNeutrals");
//			atSel2.oneVsAll("isear_turney_tfidf_testJoy");
//		} catch (IOException e1) {
//			 //TODO Auto-generated catch block
//			System.out.println("Exception - in main selection load file");
//			e1.printStackTrace();
//		}
//		
//		atSel.RankerCrossValidation(10, 1, "fairytale_balance_tfidf");
//		atSel.RankerCrossValidation(10, 2);
//		atSel.RankerCrossValidation(10, 5);
//		
//		atSel1.CfsCrossValidation(10, 1, "movies_1w_tfidf", "");
//		atSel2.CfsCrossValidation(10, 1, "movies_1w_binary", "");
//		atSel3.CfsCrossValidation(10, 1, "isear_turney_tfidf", "");
//		atSel4.CfsCrossValidation(10, 1, "isear_turney_binary", "");
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
