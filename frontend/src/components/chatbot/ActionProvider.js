import React from 'react';
import axios from 'axios';

const ActionProvider = ({ createChatBotMessage, setState, children, state }) => {
  // Define the actions that the chatbot will handle

  const handleUserMessage = async (message) => {
    const currentUrl = window.location.href;
   
    const experimentIdMatch = currentUrl.match(/\/experiment\/(\d+)/);
    const experimentId = experimentIdMatch ? experimentIdMatch[1] : null;
    console.log("Experiment ID in ActionProvider:", experimentId);
    try {
      const response = await axios.post(`http://localhost:5001/chat/${experimentId}`, {
        input: message,
      });

      const botMessage = createChatBotMessage(response.data.response);
      updateChatbotState(botMessage);
    } catch (error) {
      console.error('Error communicating with backend:', error);
      const errorMessage = createChatBotMessage('Sorry, there was an error processing your request.');
      updateChatbotState(errorMessage);
    }
  };

  // Update chatbot state with new messages
  const updateChatbotState = (message) => {
    setState((prevState) => ({
      ...prevState,
      messages: [...prevState.messages, message],
    }));
  };

  // Clone the children and pass the actions down to them
  return (
    <div>
      {React.Children.map(children, (child) => {
        return React.cloneElement(child, {
          actions: {
            handleUserMessage,  // Pass down handleUserMessage as an action
          },
        });
      })}
    </div>
  );
};

export default ActionProvider;
