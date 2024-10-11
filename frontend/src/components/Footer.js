import React from 'react';
import { Link } from 'react-router-dom';
import './Footer.css';

function Footer() {
  return (
    <>
      <footer className="footer">
        <div className="footer-container">
          <div className="footer-item">
            <img src={require('../images/iavLogo.png')} alt="Logo" className="iav-logo"/>
            <Link to='/contacts' className='footer-links'>
            Contacts
            </Link>
          </div>
        </div>
      </footer>
    </>
  );
}

export default Footer;
