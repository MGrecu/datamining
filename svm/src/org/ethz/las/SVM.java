package org.ethz.las;

import java.util.*;

public class SVM {

  static double MIN_SCALING_FACTOR = 0.0000001;

  // Hyperplane weights.
  RealVector weights;

  public SVM(RealVector weights) {
    this.weights = weights;
  }

  /**
   * Instantiates an SVM from a list of training instances, for a given
   * learning rate 'eta' and regularization parameter 'lambda'.
   */
  public static SVM createSVMSimpleOnline(List<TrainingInstance> trainingSet, double lambda) {
	  int dim = trainingSet.get(0).getFeatures().getDimension();
	  SVM svm = new SVM(new RealVector(dim));
	  int t = 0;

	  for (TrainingInstance ti : trainingSet) {
		  t++;
		  if (svm.weights.dotProduct(ti.getFeatures()) * ti.getLabel() < 1) {
			  svm.weights.add(ti.getFeatures().scale((1.0/Math.sqrt(t)) * ti.getLabel()));
			  double factor = Math.min(1.0, 1.0 / (svm.weights.getNorm() * lambda));
			  svm.weights.scaleThis(factor);
		  }
	  }
	  
	  return svm;
  }
  
  /**
   * Simple PEGASOS, found here: http://bickson.blogspot.ch/2012/04/more-on-large-scale-svm.html
   */
  public static SVM createSVMSimplePegasos(List<TrainingInstance> trainingSet, double lambda, double epochs) {
	  int dim = trainingSet.get(0).getFeatures().getDimension();
	  SVM svm = new SVM(new RealVector(dim));

      int t = 1;
	  int tLast = 1;
	  double factor = 1;
	  double eta;
	  
	  for (int i=0; i<epochs; i++) {
		  for (TrainingInstance ti: trainingSet) {
			  t++;

			  if ((svm.weights.dotProduct(ti.getFeatures()) * ti.getLabel()) < 1) {
				  factor = tLast * 1.0 / t;
				  eta = 1.0/(lambda * t);
				  svm.weights.scaleThis(factor);
				  svm.weights.add(ti.getFeatures().scale(eta * ti.getLabel())); 
				  tLast = t;
			  }
		  }
	  }
	  
	  return svm;
  }
  
  /**
   * Simple PEGASOS, with random sampling at each step
   */
//  public static SVM createSVMSimplePegasosRandom(List<TrainingInstance> trainingSet, double lambda, int epochs) {
//	  int dim = trainingSet.get(0).getFeatures().getDimension();
//	  SVM svm = new SVM(new RealVector(dim));
//	  int t = 1;
//	  int tLast = 1;
//	  double factor = 1;
//	  double eta;
//	  int N = trainingSet.size();
//	  int T = N * epochs;
//	  int maxSampling = epochs + 1;
//	  int[] counts = new int[N];
//	  double fpBias = 2;
//	  
//	  Random randomGen = new Random(123456);
//	  
//	  for (int i=0; i<T; i++) {
//		  int pos = 0;
//		  
//		  do {
//			  pos = randomGen.nextInt(N);
//		  } 
//		  while (counts[pos] >= maxSampling);
//		  
//		  counts[pos]++;
//		  TrainingInstance ti = trainingSet.get(pos);
//		  t++;
//
//		  if ((svm.weights.dotProduct(ti.getFeatures()) * ti.getLabel()) < 1) {
//			  factor = tLast * 1.0 / t;
//			  eta = 1.0/(lambda * t);
//			  svm.weights.scaleThis(factor);
//			  
//			  if (ti.getLabel() == -1) {
//				  fpBias = 2;
//			  }
//			  else {
//				  fpBias = 1.5;
//			  }
//			  
//			  svm.weights.add(ti.getFeatures().scale(fpBias * eta * ti.getLabel())); 
//			  tLast = t;
//		  }
//	  }
//	  
//	  return svm;
//  }
  
  /**
   * Simple PEGASOS, with list shuffling
   */
  public static SVM createSVMSimplePegasosShuffle(List<TrainingInstance> trainingSet, double lambda, int epochs) {
	  int dim = trainingSet.get(0).getFeatures().getDimension();
	  SVM svm = new SVM(new RealVector(dim));
	  int t = 1;
	  int tLast = 1;
	  double factor = 1;
	  double eta;
	  double fpBias = 2;
	  
	  Random randomGen = new Random(12345);
	  
	  for (int i=0; i<epochs; i++) {
		  
		  Collections.shuffle(trainingSet, randomGen);

		  for (TrainingInstance ti: trainingSet) {
			  t++;
	
			  if ((svm.weights.dotProduct(ti.getFeatures()) * ti.getLabel()) < 1) {
				  factor = tLast * 1.0 / t;
				  eta = 1.0/(lambda * t);
				  svm.weights.scaleThis(factor);
				  
				  if (ti.getLabel() == -1) {
					  fpBias = 3;
				  }
				  else {
					  fpBias = 1.5;
				  }
				  
				  svm.weights.add(ti.getFeatures().scale(fpBias * eta * ti.getLabel()));
				  tLast = t;
			  }
		  }
	  }

	  return svm;
  }
  
  /**
   * Simple PEGASOS, with balanced list shuffling
   */
  public static SVM createSVMSimplePegasosBalanced(List<TrainingInstance> trainingSet, double lambda) {
	  int dim = trainingSet.get(0).getFeatures().getDimension();
	  SVM svm = new SVM(new RealVector(dim));
	  double factor = 1;
	  double eta;
	  
	  List<TrainingInstance> positives = new ArrayList<TrainingInstance>();
	  List<TrainingInstance> negatives = new ArrayList<TrainingInstance>();
	  
	  int positiveSize = 0;
	  int negativeSize = 0;
	  for (int i = 0; i < trainingSet.size(); ++i) {
		  TrainingInstance crt = trainingSet.get(i);
		  if (crt.getLabel() == -1)
			  negatives.add(crt);
		  else
			  positives.add(crt);
	  }

	  positiveSize = positives.size();
	  negativeSize = negatives.size();

	  Random randomGen = new Random(123456);
	  Collections.shuffle(positives, randomGen);
	  Collections.shuffle(negatives, randomGen);
	  
	  for (int i = 2; i < Math.min(positiveSize, negativeSize); i++) {
		  List<TrainingInstance> balancedSet = new ArrayList<TrainingInstance>();
		  balancedSet.add(positives.get(i-2));
		  balancedSet.add(negatives.get(i-2));

		  for (TrainingInstance ti : balancedSet) {
			  if ((svm.weights.dotProduct(ti.getFeatures()) * ti.getLabel()) < 1) {
				  eta = 1.0 / (lambda * i);
				  factor = 1 - 1.0 / i;
				  
				  svm.weights.scale(factor);
				  svm.weights.add(ti.getFeatures().scale(eta * ti.getLabel())); 

				  svm.weights.scaleThis(Math.min(1.0, 1.0 / (Math.sqrt(lambda) * svm.weights.getNorm())));
			  }
		  }
	  }

	  return svm;
  }
  
  /**
   * Simple Pairwise Learning with PEGASOS
   */
  public static SVM createSVMPairwisePegasosRandom(List<TrainingInstance> trainingSet, double lambda, double epochs) {
	  int dim = trainingSet.get(0).getFeatures().getDimension();
	  SVM svm = new SVM(new RealVector(dim));

	  double factor = 1;
	  double eta;
	  
	  Random randomGen = new Random(123456);
	  int maxSamples = trainingSet.size();

	  for (int i = 2; i < epochs * trainingSet.size(); i++) {
		  TrainingInstance a, b;
		  do {
			  a = trainingSet.get(randomGen.nextInt(maxSamples));
			  b = trainingSet.get(randomGen.nextInt(maxSamples));
		  } while (a.getLabel() != b.getLabel());

		  TrainingInstance ti = new TrainingInstance(b.getFeatures().scale(-1),
				  a.getLabel() > b.getLabel() ? 1 : -1);
		  ti.getFeatures().add(a.getFeatures());

		  if ((svm.weights.dotProduct(ti.getFeatures()) * ti.getLabel()) < 1) {
			  factor = (i - 1) * 1.0 / i;
			  eta = 1.0 / (lambda * i);

			  svm.weights.scaleThis(factor);
			  svm.weights.add(ti.getFeatures().scale(eta * ti.getLabel())); 

			  svm.weights.scaleThis(Math.min(1.0, 1/(Math.sqrt(lambda) * svm.weights.getNorm())));
		  }
	  }

	  return svm;
  }  
  /**
   * Simple ROMMA, with list shuffling
   */
  public static SVM createSVMROMMAShuffle(List<TrainingInstance> trainingSet, double lambda, int epochs) {
	  int dim = trainingSet.get(0).getFeatures().getDimension();
	  SVM svm = new SVM(new RealVector(dim));
	  int t = 1;
	  
	  Random randomGen = new Random(123456);
	  
	  for (int i=0; i<epochs; i++) {
		  
		  Collections.shuffle(trainingSet, randomGen);

		  for (TrainingInstance ti: trainingSet) {
			  t++;
	
			  if (svm.weights.dotProduct(ti.getFeatures()) < ti.getLabel()) {
				  double x = ti.getFeatures().getNorm();
				  double w = svm.weights.getNorm();
				  double x2w2 = x * x * w * w;
				  double xw = ti.getFeatures().dotProduct(svm.weights);
				  double divideBy = x2w2 - xw * xw;
				  
				  double c = (x2w2 - ti.getLabel() * xw) / divideBy;
				  double d = w * w * (ti.getLabel() - xw) / divideBy;
				  
				  svm.weights.scaleThis(c);
				  svm.weights.add(ti.getFeatures().scale(d));
			  }
		  }
	  }
	  
	  return svm;
  }
    
  /**
   * Bagging with anything
   */
  public static SVM createSVMPegasosMinibatch(List<TrainingInstance> trainingSet, int minibatchSize, double lambda, int epochs)
  {
	  int dim = trainingSet.get(0).getFeatures().getDimension();
	  
	  SVM svm = new SVM(new RealVector(dim));
	  int maxSampleIndex = trainingSet.size();
	  
	  // Build the minibatch!
	  int T = epochs;
	  Random rand = new Random(123451);
	  
	  for (int t = 2; t <= T; t++) {
		  List<TrainingInstance> trainingMinibatch = new ArrayList<TrainingInstance>();
		  for (int sample = 0; sample < minibatchSize; ++sample) {
			  // Sample with replacement (e.g. same sample can end up multiple times)
			  int randIndex = rand.nextInt(maxSampleIndex);
			  trainingMinibatch.add(trainingSet.get(randIndex));
		  }

		  RealVector partialGradient = new RealVector(dim);
		  for (TrainingInstance ti : trainingMinibatch)
			  if (ti.getLabel() * ti.getFeatures().dotProduct(svm.weights) < 1) {
				  partialGradient.add(ti.getFeatures().scale(ti.getLabel()));
			  }

		  double eta = 1.0 / (lambda * t);
		  
		  /*
		   * As implemented by VLFEAT
		   */
		  // partialGradient.scaleThis(-1.0 / (eta * t));
		  
		  /*
		   * As implemented in the original paper 
		   */
		  partialGradient.scaleThis(eta/minibatchSize);
		  svm.weights.scaleThis(1 - 1.0/t);
		  
		  svm.weights.add(partialGradient);
		  
		  /*
		   * As implemented by VLFEAT
		   */
		  // svm.weights.scaleThis(Math.min(1.0, (Math.sqrt(lambda) / svm.weights.getNorm())));

		  /*
		   * As implemented in the original paper 
		   */
		  svm.weights.scaleThis(Math.min(1.0, 1 / (Math.sqrt(lambda) * svm.weights.getNorm())));

		  /*
		   * Validation isn't really anywhere in papers, but keeping the code here a while.
		  double previousMisclassifiedNegativesPrctg = 1.0;
	  	  RealVector previousWeights = new RealVector(dim);

  		  int currentMisclassifiedNegatives = 0;
		  int totalNegativesInMinibatch = 0;
		  
		  for (int sample = 0; sample < minibatchSize / 30; ++sample) {
			  int randIndex = rand.nextInt(maxSampleIndex);
			  TrainingInstance test = trainingSet.get(randIndex);

			  if (test.getLabel() == -1) {
				  totalNegativesInMinibatch++;
				  if (test.getLabel() == -1 && test.getLabel() * test.getFeatures().dotProduct(svm.weights) < 1)
					  currentMisclassifiedNegatives++;
			  }
		  }

		  double currentMisclassifiedNegativesPrctg =
			  (double) currentMisclassifiedNegatives / totalNegativesInMinibatch;

		  if (currentMisclassifiedNegatives > previousMisclassifiedNegativesPrctg) {
			  svm.weights = previousWeights;
			  t--;
		  } else {
			  previousMisclassifiedNegativesPrctg = currentMisclassifiedNegativesPrctg;
			  previousWeights = svm.weights;
		  }
		  */
	  }

	  return svm;
  }
  /**
   * PEGASOS with Bootstrapping as in the paper
   */
  public static SVM createSVMBatchPegasos(List<TrainingInstance> trainingSet, double lambda, int T, int minibatchSize)
  {
	  int dim = trainingSet.get(0).getFeatures().getDimension();
	  SVM svm = new SVM(new RealVector(dim));
	  int maxSampleIndex = trainingSet.size();
	  
	  for (int t = 1; t <= T; t++) {
		  List<TrainingInstance> minibatch = new ArrayList<TrainingInstance>();

		  // Build the minibatch!
		  for (int sample = 0; sample < minibatchSize; ++sample) {
			  int randIndex;
			  // do {
				  randIndex = (int) (Math.random() * maxSampleIndex);
			  // } while (minibatch.contains(trainingSet.get(randIndex)));
			  minibatch.add(trainingSet.get(randIndex));
		  }
		  
		  // Compute weight adjustment value.
		  RealVector gradient = new RealVector(dim);
		  for (int sample = 0; sample < minibatchSize; sample++) {
			  TrainingInstance instance = minibatch.get(sample);
			  if (instance.getLabel() * instance.getFeatures().dotProduct(svm.weights) < 1) {
				  gradient.add(instance.getFeatures().scale(instance.getLabel()));
			  }
		  }

		  // Compute eta and include it in the gradient computations.
		  double eta = 1.0/(t*lambda);
		  gradient.scaleThis(eta / minibatchSize);
		  gradient.add(svm.weights.scale(1 - eta * Math.sqrt(lambda)));

		  double finalScaleFactor = 1.0 / (gradient.getNorm() * Math.sqrt(lambda));
		  if (finalScaleFactor < 1)
			  gradient.scaleThis(finalScaleFactor);

		  svm.weights = new RealVector(gradient.w);
	  }
	  
	  return svm;
  }

  /**
   * Instantiates SVM from weights given as a string.
   */
  public SVM(String w) {
    List<Double> ll = new LinkedList<Double>();
    Scanner sc = new Scanner(w);
    while(sc.hasNext()) {
      double coef = sc.nextDouble();
      ll.add(coef);
    }
    
    sc.close();

    double[] weights = new double[ll.size()];
    int cnt = 0;
    for (Double coef : ll)
      weights[cnt++] = coef;

    this.weights = new RealVector(weights);
  }

  /**
   * Instantiates the SVM model as the average model of the input SVMs.
   */
  public SVM(List<SVM> svmList) {
    int dim = svmList.get(0).getWeights().getDimension();
    RealVector weights = new RealVector(dim);
    for (SVM svm : svmList)
      weights.add(svm.getWeights());

    this.weights = weights.scaleThis(1.0/svmList.size());
  }

  /**
   * Given a training instance it returns the result of sign(weights'instanceFeatures).
   */
  public int classify(TrainingInstance ti) {
    double result = ti.getFeatures().dotProduct(this.weights);
    if (result > 0) return 1;
    else return -1;
  }

  public RealVector getWeights() {
    return this.weights;
  }

  @Override
  public String toString() {
    return weights.toString();
  }
}
