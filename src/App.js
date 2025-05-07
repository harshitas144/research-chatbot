import React, { useState, useEffect, useRef } from 'react';
import './App.css';

// Component for displaying a single message in the chat
const ChatMessage = ({ message, isUser }) => {
  return (
    <div className={`message ${isUser ? 'user-message' : 'bot-message'}`}>
      <div className="message-avatar">
        {isUser ? '👤' : '🤖'}
      </div>
      <div className="message-content">
        <div className="message-text">{message.text}</div>
        {message.papers && message.papers.length > 0 && (
          <div className="relevant-papers">
            <h4>Relevant Papers:</h4>
            <ul>
              {message.papers.map((paper, index) => (
                <li key={index}>
                  <strong>{paper.title}</strong>
                  {paper.problem && <p><em>Problem:</em> {paper.problem.substring(0, 100)}...</p>}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};

// Component for topic selection
const TopicSelector = ({ topics, onSelectTopic }) => {
  return (
    <div className="topic-selector">
      <h3>Research Topics</h3>
      <div className="topics-list">
        {topics.map((topic) => (
          <div 
            key={topic.id}
            className="topic-item"
            onClick={() => onSelectTopic(topic)}
          >
            <div className="topic-name">{topic.name}</div>
            <div className="topic-count">{topic.count} papers</div>
          </div>
        ))}
      </div>
    </div>
  );
};

function App() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [topics, setTopics] = useState([]);
  const [loading, setLoading] = useState(false);
  const [initialized, setInitialized] = useState(false);
  const messageEndRef = useRef(null);

  // API base URL - change this to your backend URL
  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

  // Fetch topics on component mount
  useEffect(() => {
    const fetchTopics = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/topics`);
        if (response.ok) {
          const data = await response.json();
          setTopics(data.topics);
          
          // Add welcome message only once
          if (!initialized) {
            setMessages([
              { 
                text: "Welcome to the Research Paper Chatbot! I can help you find information about research papers. You can ask me questions or explore papers by topic.",
                isUser: false
              }
            ]);
            setInitialized(true);
          }
        } else {
          console.error('Failed to fetch topics');
        }
      } catch (error) {
        console.error('Error fetching topics:', error);
      }
    };

    fetchTopics();
  }, [API_BASE_URL, initialized]);

  // Scroll to bottom of chat on new messages
  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Handle sending a message to the chatbot
  const handleSendMessage = async () => {
    if (!input.trim()) return;

    // Add user message to chat
    const userMessage = { text: input, isUser: true };
    setMessages(prevMessages => [...prevMessages, userMessage]);

    // Clear input field
    setInput('');
    setLoading(true);

    try {
      // Send query to backend
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: input }),
      });

      if (response.ok) {
        const data = await response.json();
        
        // Add bot response to chat
        setMessages(prevMessages => [
          ...prevMessages, 
          { 
            text: data.response, 
            isUser: false,
            papers: data.relevant_papers 
          }
        ]);
      } else {
        // Handle error
        setMessages(prevMessages => [
          ...prevMessages, 
          { 
            text: "Sorry, I encountered an error processing your request.", 
            isUser: false 
          }
        ]);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prevMessages => [
        ...prevMessages, 
        { 
          text: "Sorry, there was an error communicating with the server.", 
          isUser: false 
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Handle selecting a topic
  const handleSelectTopic = async (topic) => {
    // Add user message for topic selection
    const userMessage = { text: `Show me papers about ${topic.name}`, isUser: true };
    setMessages(prevMessages => [...prevMessages, userMessage]);
    
    setLoading(true);

    try {
      // Fetch papers for the selected topic
      const response = await fetch(`${API_BASE_URL}/papers?topic_id=${topic.id}`);
      
      if (response.ok) {
        const data = await response.json();
        
        // Format response with paper information
        let responseText = `Here are some papers about ${topic.name}:\n\n`;
        
        if (data.papers && data.papers.length > 0) {
          // Add bot response to chat
          setMessages(prevMessages => [
            ...prevMessages, 
            { 
              text: responseText, 
              isUser: false,
              papers: data.papers.slice(0, 5) // Show top 5 papers for brevity
            }
          ]);
        } else {
          setMessages(prevMessages => [
            ...prevMessages, 
            { 
              text: `No papers found for topic ${topic.name}.`, 
              isUser: false 
            }
          ]);
        }
      } else {
        // Handle error
        setMessages(prevMessages => [
          ...prevMessages, 
          { 
            text: "Sorry, I encountered an error fetching papers for this topic.", 
            isUser: false 
          }
        ]);
      }
    } catch (error) {
      console.error('Error fetching papers:', error);
      setMessages(prevMessages => [
        ...prevMessages, 
        { 
          text: "Sorry, there was an error communicating with the server.", 
          isUser: false 
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Handle pressing Enter key to send message
  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Research Paper Chatbot</h1>
      </header>
      
      <div className="container">
        <div className="sidebar">
          <TopicSelector topics={topics} onSelectTopic={handleSelectTopic} />
        </div>
        
        <div className="chatbot-container">
          <div className="chat-messages">
            {messages.map((message, index) => (
              <ChatMessage key={index} message={message} isUser={message.isUser} />
            ))}
            {loading && (
              <div className="message bot-message">
                <div className="message-avatar">🤖</div>
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messageEndRef} />
          </div>
          
          <div className="chat-input">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask a question about research papers..."
              disabled={loading}
            />
            <button onClick={handleSendMessage} disabled={loading || !input.trim()}>
              {loading ? '...' : 'Send'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;