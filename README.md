# MEDIA-AI: Misinformation Eradication using AI-Driven Intelligent Algorithms

## Introduction 📖

MEDIAi is an innovative AI-driven project aimed at combating the growing problem of misinformation across various platforms, including live broadcasted news, social media, and online articles. Designed to be a comprehensive misinformation detection tool, MEDIAi uses cutting-edge machine learning and natural language processing (NLP) techniques to analyze and flag false claims in real time.

Currently, MEDIAi is **optimized for analyzing articles**, allowing it to detect inaccuracies and provide fact-based insights. While the system is already effective in processing and verifying text-based content from articles, its capabilities for live broadcasts and social media platforms are **still under development**.

This phased approach ensures that MEDIAi is built on a strong foundation, with new features being added incrementally to enhance its performance and reach. By focusing on scalability and precision, MEDIAi aims to become a powerful tool in the fight against misinformation in the digital age.

## Features

- Real-time misinformation detection in:
  - Live channel broadcasts with instant alerts
  - YouTube videos (transcript and caption analysis)
  - Social media posts
  - News articles
- Interactive dashboard with graphical representation of misinformation trends
- Cross-reference checking with trusted sources
- Multi-language support
- Confidence scoring system

## Prerequisites

What things you need to install the software and how to install them:

1. Python 3.6

- This setup requires that your machine has python 3.6 installed on it. you can refer to this url https://www.python.org/downloads/ to download python. Once you have python downloaded and installed, you will need to setup PATH variables (if you want to run python program directly, detail instructions are below in *how to run software section*). To do that check this: https://www.pythoncentral.io/add-python-to-path-python-is-not-recognized-as-an-internal-or-external-command/.
- Setting up PATH variable is optional as you can also run program without it and more instruction are given below on this topic.

2. Second and easier option is to download anaconda and use its anaconda prompt to run the commands. To install anaconda check this url https://www.anaconda.com/download/
3. You will also need to download and install below 3 packages after you install either python or anaconda from the steps above

- Sklearn (scikit-learn)
- numpy
- scipy
- if you have chosen to install python 3.6 then run below commands in command prompt/terminal to install these packages

  ```
  pip install -U scikit-learn
  pip install numpy
  pip install scipy
  ```

  - if you have chosen to install anaconda then run below commands in anaconda prompt to install these packages

  ```
  conda install -c scikit-learn
  conda install -c anaconda numpy
  conda install -c anaconda scipy
  ```

## Working Principle

```
User Input → Preprocessing → Misinformation Detection → Cross-Checking → Data Visualization → User Output
```

## Technical Stack

### Frontend (User Interface)

- React.js with Redux for state management
- Bootstrap for responsive design
- Chart.js and D3.js for data visualization

### Backend (Server and API)

- Python (Flask/Django)
- Node.js for real-time features
- GraphQL for API optimization

### Machine Learning & NLP

- TensorFlow and Keras
- spaCy and NLTK
- Hugging Face Transformers
- OpenAI GPT

### Audio/Video Processing

- Google Speech-to-Text API
- FFmpeg
- OpenCV

### Real-Time Data Processing

- Apache Kafka
- Apache Spark
- Celery

### Database

- MongoDB for unstructured data
- PostgreSQL for structured data
- Elasticsearch for text search

### Cloud Infrastructure & Deployment

- AWS (EC2, S3, SageMaker, Lambda)
- Docker and Kubernetes
- Google Cloud Platform

### Version Control & Collaboration

- Git
- GitHub/GitLab

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/media-ai.git

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
```

## Usage

```python
# Start the detection service
python src/main.py

# Access the dashboard
http://localhost:3000
```

## Data Pipeline

1. **Data Collection**

   - Ministry of Information and Broadcasting datasets
   - FakeNewsNet
   - LIAR Dataset
   - Fact-checking APIs integration
2. **Preprocessing**

   - Text tokenization
   - Noise removal
   - Entity recognition
   - Metadata annotation
3. **Detection Process**

   - Knowledge graph analysis
   - Pattern recognition
   - Context evaluation
   - Source verification

## Testing

```bash
# Run unit tests
python -m pytest tests/

# Validation metrics
python scripts/validate.py
```

## File descriptions

#### DataPrep.py

This file contains all the pre processing functions needed to process all input documents and texts. First we read the train, test and validation data files then performed some pre processing like tokenizing, stemming etc. There are some exploratory data analysis is performed like response variable distribution and data quality checks like null or missing values etc.

#### FeatureSelection.py

In this file we have performed feature extraction and selection methods from sci-kit learn python libraries. For feature selection, we have used methods like simple bag-of-words and n-grams and then term frequency like tf-tdf weighting. we have also used word2vec and POS tagging to extract the features, though POS tagging and word2vec has not been used at this point in the project.

#### classifier.py

Here we have build all the classifiers for predicting the fake news detection. The extracted features are fed into different classifiers. We have used Naive-bayes, Logistic Regression, Linear SVM, Stochastic gradient descent and Random forest classifiers from sklearn. Each of the extracted features were used in all of the classifiers. Once fitting the model, we compared the f1 score and checked the confusion matrix. After fitting all the classifiers, 2 best performing models were selected as candidate models for fake news classification. We have performed parameter tuning by implementing GridSearchCV methods on these candidate models and chosen best performing parameters for these classifier. Finally selected model was used for fake news detection with the probability of truth. In Addition to this, We have also extracted the top 50 features from our term-frequency tfidf vectorizer to see what words are most and important in each of the classes. We have also used Precision-Recall and learning curves to see how training and test set performs when we increase the amount of data in our classifiers.

#### prediction.py

Our finally selected and best performing classifier was ``Logistic Regression`` which was then saved on disk with name ``final_model.sav``. Once you close this repository, this model will be copied to user's machine and will be used by prediction.py file to classify the fake news. It takes an news article as input from user then model is used for final classification output that is shown to user along with probability of truth.

Below is the Process Flow of the project:

<p align="center">
  <img width="600" height="750" src="https://github.com/sujansaitej/MediAi/blob/Product-MediAi/images/ProcessFlow.PNG">
</p>

### Performance

Below is the learning curves for our candidate models.

**Logistic Regression Classifier**

<p align="center">
  <img width="550" height="450" src="https://github.com/nishitpatel01/Fake_News_Detection/blob/master/images/LR_LCurve.PNG">
</p>

**Random Forest Classifier**

<p align="center">
  <img width="550" height="450" src="https://github.com/sujansaitej/MediAi/blob/Product-MediAi/images/RF_LCurve.png">
</p>

## Future Updates 📢

In the near future, MEDIAi will expand its capabilities to detect misinformation across a wider range of platforms, including **live broadcasted news** and **social media posts**. This enhancement will allow the model to analyze dynamic, real-time content and flag false claims as they emerge, ensuring timely and accurate fact-checking for a broader audience.

Our team is actively working on integrating advanced real-time processing and NLP techniques to make these features robust and scalable. Once implemented, MEDIAi will serve as a comprehensive solution for combating misinformation across all major media platforms, further solidifying its impact in promoting truthful and transparent communication. Stay tuned for updates as we move closer to achieving this milestone!

## Conclusion 📄

MEDIAi is a promising demo project designed to tackle the growing issue of misinformation in the digital age. While the current version focuses on analyzing articles for false claims and inaccuracies, the potential for this system extends far beyond its current capabilities. With future updates aimed at expanding the model's ability to detect misinformation in **live broadcasted news**, **social media posts**, and other dynamic content, MEDIAi is on track to become a comprehensive solution for real-time misinformation detection across all media platforms.

As the accuracy of the model continues to improve, we aim to provide a more reliable, scalable, and efficient tool that enhances transparency and accountability in information dissemination. The development of MEDIAi marks an important step toward fostering a more informed society, and we are excited to see how it evolves in the future. Stay tuned for further advancements, and thank you for exploring the demo version of MEDIAi!

## Contributors

Team of students from Saveetha Institute of Medical and Technical Sciences:

- Sujan Saitej L (Team Leader)
- S. Mohamed Nasir
- Aathif S Basha
- Rejolin Solomon J

## Contact

Sujan Saitej L
Email: sujansaitej07@gmail.com
Phone: +91 93618 60665

## Thank You 🙏
