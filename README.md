# QuranicSentiment 🕌 ✨

An intelligent web application that provides relevant Quranic verses based on emotional states, combining sentiment analysis with spiritual guidance.

🌐 **Live Site**: [https://ymorsi7.github.io/QuranicSentiment/](https://ymorsi7.github.io/QuranicSentiment/)

**NOTE: This is NOT a ChatGPT wrapper 😂**

![QuranicSentiment Interface](docs/page.png)

## Features 🌟

- **Emotion Detection** 🎭
  - Real-time analysis of user-inputted emotions
  - Support for both direct emotion selection and free-text input
  - Advanced sentiment analysis using VADER and TextBlob
  - Six primary emotional categories: Joyful, Peaceful, Fearful, Angry, Remorseful, and Reflective

- **Verse Matching** 📖
  - Intelligent matching of emotional states to relevant Quranic verses
  - Context-aware verse selection using both sentiment and linguistic analysis
  - Support for both exact and thematic emotional matches
  - Related verses suggestions based on content similarity

- **User Interface** 💫
  - Clean, modern, and responsive design
  - Intuitive emotion selection buttons
  - Smooth animations and transitions
  - Beautiful neumorphic styling

## Technical Stack 🛠️

- **Frontend**:
  - HTML5, CSS3, JavaScript
  - Custom CSS animations and transitions
  - Responsive design with CSS Grid and Flexbox
  - Neumorphic UI components

- **Backend**:
  - Python 3.x
  - Flask web framework
  - Natural Language Processing:
    - VADER Sentiment Analysis
    - TextBlob
    - NLTK
  - Pandas for data processing

- **Data**:
  - JSON-based Quran database
  - Structured emotional word dictionaries
  - Sentiment mapping system
 
## Machine Learning Pipeline 🧠

- Sentiment Analysis:

  - VADER scoring (neg, neu, pos, compound)

  - Custom linguistic feature extraction

  - Emotion-word dictionary mapping

- Model Performance:

  - 89% accuracy on verse classification

  - 0.87 F1 Score

  - 0.14 Balanced Error Rate

- Verse Classification:

  - Two-stage classification process

  - Multi-label emotion mapping

  - Confidence scoring system

## Data Processing 📊

- Preprocessing:

  - Tokenization and lemmatization

  - Stopword removal

  - Feature extraction including sentiment polarity

- Dataset Statistics:

  - 6,236 total verses

  - 6 emotional categories

  - Distribution: 15% Joyful, 25% Peaceful, 10% Angry, 20% Fearful, 20% Remorseful, 10% Reflective

## Installation 🚀

1. Clone the repository:
```bash
git clone https://github.com/ymorsi7/QuranicSentiment.git
cd QuranicSentiment
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage 📱

1. **Direct Emotion Selection**:
   - Click on any emotion button that matches your current feeling
   - Receive a relevant Quranic verse instantly

2. **Text Input**:
   - Type how you're feeling in the input box
   - Click "Analyze" or press Enter
   - The system will detect your emotion and provide a matching verse

3. **Related Verses**:
   - Explore similar verses shown below the main verse
   - Click on any related verse to view it

## Emotional Categories 🎭

- **Joyful** 😊: Verses about happiness, gratitude, and divine blessings
- **Peaceful** 😌: Content focusing on tranquility, security, and inner peace
- **Fearful** 😨: Verses addressing anxiety, worry, and divine protection
- **Angry** 😠: Content about patience, self-control, and forgiveness
- **Remorseful** 😔: Verses about seeking forgiveness and divine mercy
- **Reflective** 🤔: Content encouraging contemplation and understanding

## Project Structure 📁

```
QuranicSentiment/
├── docs/
│   ├── icon.png
│   ├── page.png
│   └── index.html
├── app.py
├── main.py
├── quran.json
└── README.md
```


## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- Quran data source
- VADER Sentiment Analysis
- TextBlob and NLTK communities
- All contributors and supporters

## Contact 📬

Contributors:
- Yusuf Morsi
- Younus Ahmad
- Ali Alani

Project Links:
- Repository: [https://github.com/ymorsi7/QuranicSentiment](https://github.com/ymorsi7/QuranicSentiment)
- Live Site: [https://ymorsi7.github.io/QuranicSentiment/](https://ymorsi7.github.io/QuranicSentiment/)
