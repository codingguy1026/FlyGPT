const chatMessages = document.getElementById('chatMessages');
const chatForm = document.getElementById('chatForm');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');

function appendMessage(sender, content) {
  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${sender}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = sender === 'assistant' ? '🪰' : '👤';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = content;

  msgDiv.appendChild(avatar);
  msgDiv.appendChild(bubble);
  chatMessages.appendChild(msgDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function createLoadingMessage() {
  const id = `loading-${Date.now()}`;

  const wrapper = document.createElement('div');
  wrapper.className = 'message assistant';
  wrapper.id = id;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = '🪰';

  const bubble = document.createElement('div');
  bubble.className = 'bubble loading';
  bubble.textContent = '커넥톰 데이터베이스를 쿼리 중입니다...';

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  chatMessages.appendChild(wrapper);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  return id;
}

chatForm.addEventListener('submit', async (event) => {
  event.preventDefault();

  const text = userInput.value.trim();
  if (!text) return;

  userInput.value = '';
  userInput.disabled = true;
  sendBtn.disabled = true;

  appendMessage('user', text);
  const loadingId = createLoadingMessage();

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message: text,
      }),
    });

    let data;

    try {
      data = await response.json();
    } catch {
      throw new Error(`HTTP ${response.status}`);
    }

    document.getElementById(loadingId)?.remove();

    if (!response.ok) {
      appendMessage(
        'assistant',
        `⚠️ 오류 발생\n${data.detail || '알 수 없는 오류가 발생했습니다.'}`,
      );
      return;
    }

    appendMessage('assistant', data.answer);
  } catch (error) {
    document.getElementById(loadingId)?.remove();
    appendMessage(
      'assistant',
      `⚠️ 통신 오류\n서버와 연결할 수 없습니다.\n${error.message}`,
    );
  } finally {
    userInput.disabled = false;
    sendBtn.disabled = false;
    userInput.focus();
  }
});
