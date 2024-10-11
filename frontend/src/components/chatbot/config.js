import { createChatBotMessage } from 'react-chatbot-kit';
import chatbotGif from '../../images/chatbot.gif';

const botName = "IAV Bot";

const config = {
  botName: botName,
  initialMessages: [
    createChatBotMessage(`Hello! I'm ${botName}. How can I assist you with your experiment today?`)
  ],
  customStyles: {
    botMessageBox: {
      backgroundColor: "#376B7E",
    },
    chatButton: {
      backgroundColor: "#5ccc9d",
    },
  },
  botAvatar: chatbotGif,
};

export default config;
