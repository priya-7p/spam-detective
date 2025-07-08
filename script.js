// Sample email data
const sampleEmails = [
    {
        type: 'spam',
        subject: 'URGENT: Claim Your Prize NOW!!!',
        text: 'CONGRATULATIONS!!! You have WON $1,000,000 in our EXCLUSIVE lottery! Click here NOW to claim your prize! Limited time offer expires in 24 hours! Act fast! Visit www.suspicious-link.com and enter your personal details to receive your money immediately!',
        icon: '⚠️'
    },
    {
        type: 'spam',
        subject: 'Free Money - No Strings Attached!',
        text: 'Get FREE MONEY today! No credit check required! Guaranteed approval! Click this link to get $5000 cash advance instantly! www.easy-money-scam.com - Your financial freedom is just one click away!',
        icon: '⚠️'
    },
    {
        type: 'ham',
        subject: 'Weekly Team Meeting Reminder',
        text: 'Hi team, this is a friendly reminder about our weekly meeting scheduled for tomorrow at 2 PM in the conference room. We will be discussing the quarterly results and planning for next month. Please bring your project updates.',
        icon: '🛡️'
    },
    {
        type: 'ham',
        subject: 'Your order has been shipped',
        text: 'Thank you for your recent purchase! Your order #12345 has been shipped and is on its way. You can track your package using the tracking number: ABC123456. Expected delivery date is June 15th.',
        icon: '🛡️'
    }
];

// ML utilities
class SpamClassifier {
    constructor() {
        this.spamKeywords = [
            'free', 'win', 'winner', 'congratulations', 'prize', 'money', 'cash',
            'urgent', 'limited time', 'act now', 'click here', 'guarantee',
            'no strings attached', 'exclusive', 'offer expires', 'claim now'
        ];
    }

    async preprocessText(text) {
        const steps = [];
        
        // Simulate preprocessing steps with delays
        await this.delay(300);
        steps.push("Text tokenization completed");
        
        await this.delay(400);
        steps.push("Stopwords removed (a, an, the, etc.)");
        
        await this.delay(300);
        steps.push("Text normalized and lowercased");
        
        await this.delay(400);
        steps.push("TF-IDF vectorization applied");
        
        await this.delay(300);
        steps.push("Feature extraction completed");
        
        return steps;
    }

    classifyEmail(text) {
        const textLower = text.toLowerCase();
        
        // Count suspicious features
        const suspiciousWords = this.spamKeywords.filter(keyword => 
            textLower.includes(keyword)
        );
        
        const urlCount = (text.match(/https?:\/\/[^\s]+/g) || []).length;
        const exclamationCount = (text.match(/!/g) || []).length;
        const capsWords = text.match(/[A-Z]{2,}/g) || [];
        const capsPercentage = Math.round((capsWords.join('').length / text.length) * 100);
        
        // Simple scoring algorithm
        let spamScore = 0;
        
        // Weight factors
        spamScore += suspiciousWords.length * 0.15;
        spamScore += urlCount * 0.2;
        spamScore += Math.min(exclamationCount * 0.1, 0.3);
        spamScore += Math.min(capsPercentage * 0.01, 0.2);
        
        // Additional heuristics
        if (textLower.includes('click') && textLower.includes('link')) spamScore += 0.1;
        if (textLower.includes('personal details') || textLower.includes('credit card')) spamScore += 0.2;
        if (exclamationCount >= 3) spamScore += 0.15;
        
        // Determine classification
        const isSpam = spamScore > 0.5;
        const confidence = isSpam ? 
            Math.min(0.5 + spamScore, 0.98) : 
            Math.max(0.5 - spamScore, 0.02);
        
        return {
            prediction: isSpam ? 'spam' : 'ham',
            confidence,
            features: {
                suspiciousWords,
                urlCount,
                exclamationCount,
                capsPercentage
            }
        };
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// DOM elements
const emailTextarea = document.getElementById('email-text');
const classifyBtn = document.getElementById('classify-btn');
const btnText = document.querySelector('.btn-text');
const btnLoading = document.querySelector('.btn-loading');
const preprocessingCard = document.getElementById('preprocessing-card');
const preprocessingSteps = document.getElementById('preprocessing-steps');
const resultCard = document.getElementById('result-card');
const resultContent = document.getElementById('result-content');
const sampleEmailsContainer = document.getElementById('sample-emails');

// Initialize classifier
const classifier = new SpamClassifier();

// Event listeners
classifyBtn.addEventListener('click', handleClassify);

// Initialize sample emails
function initializeSampleEmails() {
    sampleEmailsContainer.innerHTML = '';
    
    sampleEmails.forEach((email, index) => {
        const emailElement = document.createElement('div');
        emailElement.className = 'sample-email';
        emailElement.innerHTML = `
            <div class="sample-header">
                <span class="sample-type ${email.type}">${email.type.toUpperCase()}</span>
                <span>${email.icon}</span>
            </div>
            <h4 class="sample-subject">${email.subject}</h4>
            <p class="sample-preview">${email.text.substring(0, 80)}...</p>
            <button class="btn-sample" onclick="selectSample(${index})">Test This Email</button>
        `;
        sampleEmailsContainer.appendChild(emailElement);
    });
}

// Handle classification
async function handleClassify() {
    const text = emailTextarea.value.trim();
    if (!text) return;
    
    // Show loading state
    classifyBtn.disabled = true;
    btnText.style.display = 'none';
    btnLoading.style.display = 'inline';
    
    // Hide previous results
    resultCard.style.display = 'none';
    preprocessingCard.style.display = 'none';
    preprocessingSteps.innerHTML = '';
    
    try {
        // Show preprocessing steps
        preprocessingCard.style.display = 'block';
        const steps = await classifier.preprocessText(text);
        
        steps.forEach((step, index) => {
            setTimeout(() => {
                const stepElement = document.createElement('div');
                stepElement.className = 'preprocessing-step';
                stepElement.innerHTML = `
                    <span class="step-badge">Step ${index + 1}</span>
                    <span>${step}</span>
                `;
                preprocessingSteps.appendChild(stepElement);
            }, index * 100);
        });
        
        // Simulate ML processing delay
        await classifier.delay(2000);
        
        // Get classification result
        const result = classifier.classifyEmail(text);
        displayResult(result);
        
    } finally {
        // Reset button state
        classifyBtn.disabled = false;
        btnText.style.display = 'inline';
        btnLoading.style.display = 'none';
    }
}

// Display classification result
function displayResult(result) {
    const isSpam = result.prediction === 'spam';
    const confidencePercentage = Math.round(result.confidence * 100);
    
    resultCard.className = `card ${isSpam ? 'spam-result' : 'ham-result'}`;
    
    resultContent.innerHTML = `
        <div class="result-header">
            <span style="font-size: 1.5rem;">${isSpam ? '❌' : '✅'}</span>
            <span class="result-badge ${result.prediction}">
                ${isSpam ? 'SPAM' : 'HAM (Safe)'}
            </span>
        </div>
        
        <div class="confidence-section">
            <div class="confidence-header">
                <span style="font-weight: 600;">Confidence Score</span>
                <span style="font-weight: 700;">${confidencePercentage}%</span>
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill ${result.prediction}" style="width: ${confidencePercentage}%"></div>
            </div>
        </div>
        
        <div class="features-grid">
            <div class="feature-section">
                <h4>Feature Analysis</h4>
                <div class="feature-list">
                    <div>URLs detected: ${result.features.urlCount}</div>
                    <div>Exclamation marks: ${result.features.exclamationCount}</div>
                    <div>CAPS percentage: ${result.features.capsPercentage}%</div>
                </div>
            </div>
            
            ${result.features.suspiciousWords.length > 0 ? `
                <div class="feature-section">
                    <h4>Suspicious Words</h4>
                    <div class="suspicious-words">
                        ${result.features.suspiciousWords.slice(0, 3).map(word => 
                            `<span class="word-badge">${word}</span>`
                        ).join('')}
                    </div>
                </div>
            ` : ''}
        </div>
    `;
    
    resultCard.style.display = 'block';
}

// Select sample email
function selectSample(index) {
    const email = sampleEmails[index];
    emailTextarea.value = email.text;
    
    // Hide previous results
    resultCard.style.display = 'none';
    preprocessingCard.style.display = 'none';
    preprocessingSteps.innerHTML = '';
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    initializeSampleEmails();
});
