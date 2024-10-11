import React, {useState, useEffect} from 'react';
import {Link} from 'react-router-dom';
import {Button} from './Button';
import './Navbar.css';
import nasaPic from '../images/nasaLogo.png'


function Navbar(){
    const [click, setClick] = useState(false);
    const [button, setButton] = useState(true)

    const handleClick = () => setClick(!click);
    const closeMobileMenu = () => setClick(false)

    const showButton = () => {
        if(window.innerWidth <= 960){
            setButton(false);
        }else{
            setButton(true);
        }
    };

    useEffect(() => {
        showButton();
    }, []);

    window.addEventListener('resize', showButton);

    return (
        <>
            <nav className="navbar">
                <div className="navbar-container">
                    
                    <Link to='/' className='nasaLogo' style = {{textDecoration: 'none'}} onClick={closeMobileMenu}>
                    <div className="nasa-logo">
                    <span className="nasa-text" style={{textDecoration: 'none'}}>NASA</span>
                        <img src ={nasaPic} alt="Nasa"style={{ width: '100px', height: 'auto' }}/>
                    </div>
                    </Link>
                    
                    <div className='menu-icon' onClick={handleClick}>
                        <i className={click ? 'fas fa-times' : 'fa-solid fa-bars'} />
                    </div>
                    <ul className = {click ? 'nav-menu active' : 'nav-menu'}>
                        <li className='nav-item'>
                            <Link to='/about' className='nav-links' onClick={closeMobileMenu}>
                                About
                            </Link>
                        </li>
                    </ul>
                </div>
            </nav>
        </>
    );
}

export default Navbar;