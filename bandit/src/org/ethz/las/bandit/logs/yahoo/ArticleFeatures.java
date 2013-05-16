package org.ethz.las.bandit.logs.yahoo;

public class ArticleFeatures {

  private int id;
  private double[] features;

  public ArticleFeatures(int id, double[] features) {
    this.id = id;
    this.features = features;
  }

  public int getID() {
    return id;
  }

  public double[] getFeatures() {
    return features;
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof Article)
      return ((Article) o).getID() == id;
    return false;
  }
}
