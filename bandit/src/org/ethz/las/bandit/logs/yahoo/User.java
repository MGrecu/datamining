package org.ethz.las.bandit.logs.yahoo;

public class User {

  private long timestamp;
  private double[] features;

  public User(long timestamp, double[] features) {
    this.features = features;
    this.timestamp = timestamp;
  }

  public long getTimestamp() {
    return timestamp;
  }

  public double[] getFeatures() {
    return features;
  }

  @Override
  public String toString() {
    return String.format("%s %.3f %.3f %.3f %.3f %.3f %.3f", timestamp, features[0], features[1],
                  features[2], features[3],features[4], features[5]);
  }
}
