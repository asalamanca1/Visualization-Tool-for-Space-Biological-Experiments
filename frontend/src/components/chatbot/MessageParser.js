import React from 'react';

const MessageParser = ({ children, actions }) => {
  const parse = (message) => {
    if (message.trim() !== "") {
      // Call the appropriate action, for example:
      actions.handleUserMessage(message);
    }
  };

  return (
    <div>
      {React.Children.map(children, (child) => {
        return React.cloneElement(child, {
          parse: parse, // Pass the parse function to the child
          actions: actions, // Pass the actions to the child
        });
      })}
    </div>
  );
};

export default MessageParser;
