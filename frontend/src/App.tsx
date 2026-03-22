import { useState, useRef, useEffect } from 'react';
import { IconButton, TextField } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import { streamMessage } from './services/chatService';
import './App.scss';

interface Message {
  role: 'user' | 'ai';
  content: string;
  formattedContent?: string;
}

function formatMarkdownAndCitations(text: string): string {
  let html = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\n\n/g, '</p><p>');
  html = '<p>' + html + '</p>';
  const citationRegex = /(\([^)]+\))\s*\[PMID:\s*(\d+)\]/g;
  html = html.replace(citationRegex, (_match, authorYear: string, pmid: string) => {
    const url = `https://pubmed.ncbi.nlm.nih.gov/${pmid}/`;
    return `<a href="${url}" target="_blank" rel="noopener noreferrer" class="pmid-link">${authorYear}</a>`;
  });
  return html;
}

const EMPTY_AI_MSG: Message = { role: 'ai', content: '', formattedContent: '' };

const welcomeMsg: Message = {
  role: 'ai',
  content: 'Hello! I am your GLP-1 & Weight Loss research assistant. Ask me anything about GLP-1 medications (Ozempic, Wegovy, Mounjaro), their side effects, safety, or nutrition during treatment.',
  formattedContent: 'Hello! I am your GLP-1 &amp; Weight Loss research assistant. Ask me anything about GLP-1 medications (Ozempic, Wegovy, Mounjaro), their side effects, safety, or nutrition during treatment.',
};

export default function App() {
  const [messages, setMessages] = useState<Message[]>([welcomeMsg]);
  const [userInput, setUserInput] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const [thinkingStatus, setThinkingStatus] = useState('');

  const sessionId = useRef('wl_session_' + Math.random().toString(36).substring(7));
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, thinkingStatus]);

  const sendMessage = async () => {
    if (!userInput.trim() || isThinking) return;

    const query = userInput;
    setMessages(msgs => [...msgs, { role: 'user', content: query }]);
    setUserInput('');
    setIsThinking(true);
    setThinkingStatus('Initializing...');

    let aiIndex = -1;
    setMessages(msgs => {
      aiIndex = msgs.length;
      return [...msgs, EMPTY_AI_MSG];
    });

    try {
      const stream = streamMessage(query, sessionId.current);
      let accumulatedText = '';

      for await (const event of stream) {
        if (event.type === 'clear_tokens') {
          accumulatedText = '';
          setMessages(msgs => {
            const next = [...msgs];
            next[aiIndex] = EMPTY_AI_MSG;
            return next;
          });
        } else if (event.type === 'status') {
          setThinkingStatus(event.message || 'Processing...');
        } else if (event.type === 'token') {
          setThinkingStatus('');
          accumulatedText += (event.content || '');
          setMessages(msgs => {
            const next = [...msgs];
            next[aiIndex] = {
              role: 'ai',
              content: accumulatedText,
              formattedContent: formatMarkdownAndCitations(accumulatedText),
            };
            return next;
          });
        }
      }
    } catch (err) {
      console.error(err);
      setMessages(msgs => {
        const next = [...msgs];
        next[aiIndex] = {
          role: 'ai',
          content: 'Connection Error: Unable to reach the backend.',
          formattedContent: '<strong>Connection Error:</strong> Unable to reach the backend.',
        };
        return next;
      });
    } finally {
      setIsThinking(false);
      setThinkingStatus('');
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      sendMessage();
    }
  };

  const aiAvatar = (
    <div className="avatar ai-avatar">
      <img src="/favicon.ico" alt="WeightLoss RAG" className="avatar-icon" />
    </div>
  );

  return (
    <div className="app-container">
      <header className="header">
        <div className="brand-section">
          <div className="logo-container">
            <img src="/favicon.ico" alt="WeightLoss RAG Logo" className="logo-image" />
            <div className="logo-text">WeightLoss RAG</div>
          </div>
          <div className="brand-divider" />
          <div className="subtitle">MEDICAL RESEARCH ASSISTANT</div>
        </div>
        <div className="header-links">
          <a
            href="https://github.com/ali-maraci/weightloss-rag-pubmed"
            target="_blank"
            rel="noopener noreferrer"
            className="github-link"
          >
            <svg height="24" viewBox="0 0 16 16" version="1.1" width="24" aria-hidden="true">
              <path
                fillRule="evenodd"
                d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"
              />
            </svg>
            <span>GitHub</span>
          </a>
        </div>
      </header>

      <div className="chat-container">
        <div className="chat-history" ref={scrollRef}>
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`message-wrapper ${msg.role}`}
              style={{ display: msg.role === 'ai' && !msg.content ? 'none' : 'flex' }}
            >
              {msg.role === 'ai' && msg.content && aiAvatar}
              {msg.content && (
                <div className="message-bubble">
                  {msg.role === 'user' ? (
                    msg.content
                  ) : (
                    <div dangerouslySetInnerHTML={{ __html: msg.formattedContent || msg.content }} />
                  )}
                </div>
              )}
            </div>
          ))}

          {thinkingStatus && (
            <div className="message-wrapper ai">
              {aiAvatar}
              <div className="message-bubble thinking-bubble">
                <div className="thinking-dots">
                  <span /><span /><span />
                </div>
                <span className="thinking-text">{thinkingStatus}</span>
              </div>
            </div>
          )}
        </div>

        <div className="input-area">
          <TextField
            multiline
            maxRows={6}
            minRows={1}
            placeholder="Ask about GLP-1 medications, side effects, safety, or nutrition during weight loss treatment..."
            value={userInput}
            onChange={e => setUserInput(e.target.value)}
            onKeyDown={onKeyDown}
            disabled={isThinking}
            inputRef={inputRef}
            className="chat-input"
            sx={{
              width: '100%',
              '& .MuiOutlinedInput-root': {
                borderRadius: '28px',
                backgroundColor: '#ffffff',
                boxShadow: '0 4px 20px rgba(0,0,0,0.06)',
                alignItems: 'flex-end',
                paddingLeft: '16px',
                paddingRight: '8px',
                '& fieldset': { border: 'none' },
              },
              '& .MuiInputBase-input': {
                resize: 'none',
                marginTop: '14px',
                marginBottom: '14px',
                lineHeight: 1.5,
                overflowY: 'hidden',
                '&::placeholder': { color: '#94a3b8', opacity: 1 },
              },
              '& .MuiInputBase-inputMultiline': {
                paddingTop: 0,
                paddingBottom: 0,
              },
            }}
            slotProps={{
              input: {
                endAdornment: (
                  <IconButton
                    onClick={sendMessage}
                    disabled={!userInput.trim() || isThinking}
                    color="primary"
                    sx={{ mb: '8px', flexShrink: 0 }}
                  >
                    <SendIcon />
                  </IconButton>
                ),
              },
            }}
          />
        </div>

        <footer className="app-footer">
          <p>&copy; 2026 Ali Maraci. All rights reserved.</p>
          <p>WeightLoss RAG is an experimental research tool and does not provide medical advice.</p>
        </footer>
      </div>
    </div>
  );
}
