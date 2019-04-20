from faker import Faker
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline

from ptavitm.sklearn_api import ProdLDATransformer


def test_basic():
    fake = Faker()
    fake.seed(0)
    pipeline = make_pipeline(
        CountVectorizer(
            stop_words='english',
            max_features=25,
            max_df=0.9
        ),
        ProdLDATransformer(
            epochs=1
        )
    )
    pipeline.fit([fake.text() for _ in range(100)])
    result = pipeline.transform([fake.text() for _ in range(20)])
    assert result.shape == (20, 50)


def test_score():
    fake = Faker()
    fake.seed(0)
    pipeline = Pipeline(steps=[
        ('vectorizer', CountVectorizer(
            stop_words='english',
            max_features=25,
            max_df=0.9
        )),
        ('topic', ProdLDATransformer(epochs=1))
    ])
    param_grid = {
        'topic__epochs': [1, 2, 3],
    }
    search = GridSearchCV(pipeline, param_grid, iid=False, cv=2, return_train_score=False)
    search.fit([fake.text() for _ in range(10)])
