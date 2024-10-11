import React from 'react'
import './About.css';
import CardItem from './CardItem';

function About() {
    return (
        <div className='cards'>
          <h1 style={{ fontFamily: 'Raleway, sans-serif', fontWeight: 100 }}>Thank you for using our tool!</h1>
          <div className="cards__container">
            <div className="cards__warpper">
                <ul className="cards__items">
                    <CardItem 
                    src="images/nasamoon.jpg"
                    text = "Explore the Objective"
                    label='Objective'
                    href="https://www.spaceappschallenge.org/nasa-space-apps-2024/challenges/visualize-space-science/?tab=details"
                    fontWeight ="100"
                    />
                    <CardItem 
                    src="images/logo2_2.png"
                    text = "Our team"
                    label='IAV'
                    href="https://www.spaceappschallenge.org/nasa-space-apps-2024/find-a-team/space-meatballs/?tab=members"
                    fontWeight = "100"
                    />
                </ul>
            </div>
          </div>
        </div>
    )
}

export default About
