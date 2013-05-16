package org.ethz.las.bandit.logs.yahoo;

public class Article {

	private int id;

	public Article(int id) {
		this.id = id;
	}

	public int getID() {
		return id;
	}

	@Override
	public boolean equals(Object o) {
		if (o instanceof Article)
			return ((Article) o).getID() == id;
		return false;
	} 

}
