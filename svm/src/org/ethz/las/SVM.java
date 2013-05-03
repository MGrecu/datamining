package org.ethz.las;

import java.util.*;

public class SVM {
	
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
  public static SVM createSVMSimplePegasosRandom(List<TrainingInstance> trainingSet, double lambda, int epochs) {
	  int dim = trainingSet.get(0).getFeatures().getDimension();
	  SVM svm = new SVM(new RealVector(dim));
	  int t = 1;
	  int tLast = 1;
	  double factor = 1;
	  double eta;
	  int S = trainingSet.size();
	  int T = S * epochs;
	  
	  for (int i=0; i<T; i++) {
		  Random randomGen = new Random();
		  TrainingInstance ti = trainingSet.get(randomGen.nextInt(S));
		  t++;

		  if ((svm.weights.dotProduct(ti.getFeatures()) * ti.getLabel()) < 1) {
			  factor = tLast * 1.0 / t;
			  eta = 1.0/(lambda * t);
			  svm.weights.scaleThis(factor);
			  svm.weights.add(ti.getFeatures().scale(eta * ti.getLabel())); 
			  tLast = t;
		  }
	  }
	  
	  return svm;
  }
  
  /**
   * PEGASOS with bootstrapping over existing algorithm.
   */
  public SVM(List<TrainingInstance> trainingSet, double lambda, int minibatchSize, int epochs)
  {
	  int dim = trainingSet.get(0).getFeatures().getDimension();
	  this.weights = new RealVector(dim);
	  int maxSampleIndex = trainingSet.size();
	  
	  int tLast = 1;
	  double factor = 1;
	  double eta;
	  
	  int T = 1 + (int) (Math.random() * epochs);
	  for (int t = 1; t <= T; t++) {
		  List<TrainingInstance> minibatch = new ArrayList<TrainingInstance>();

		  // Build the minibatch!
		  for (int sample = 0; sample < minibatchSize; ++sample) {
			  int randIndex;
			  do {
				  randIndex = (int) (Math.random() * maxSampleIndex);
			  } while (minibatch.contains(trainingSet.get(randIndex)));
			  minibatch.add(trainingSet.get(randIndex));
		  }
		 
		  for (TrainingInstance ti : minibatch) {			  
			  if ((weights.dotProduct(ti.getFeatures()) * ti.getLabel()) < 1) {
				  factor = tLast * 1.0 / t;
				  eta = 1.0/(lambda * t);
				  weights.scaleThis(factor);
				  weights.add(ti.getFeatures().scale(eta * ti.getLabel())); 
			  }
		  }		  
		  tLast = t;
	  }
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
    @SuppressWarnings("unused")
	RealVector features = ti.getFeatures();

    double result = ti.getFeatures().dotProduct(this.weights);
    if (result >= 0) return 1;
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
