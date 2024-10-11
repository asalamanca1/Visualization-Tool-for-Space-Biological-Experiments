import React from 'react';
import './Popup.css';

const Popup = ({ isVisible, onClose, title, children }) => {
  if (!isVisible) return null;

  return (
    <div className="popup-overlay" onClick={onClose}>
      <div className="popup-content" onClick={(e) => e.stopPropagation()}>
        <div className="popup-header">
          <h2>{title}</h2>
          <button className="close-button" onClick={onClose}>
            &times;
          </button>
        </div>
        <div className="popup-body">
          {children}  {/* This will render the passed overviewContent */}
        </div>
      </div>
    </div>
  );
};

export default Popup;
