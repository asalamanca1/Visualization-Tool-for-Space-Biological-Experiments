import React from 'react'
import { Link } from 'react-router-dom'

function CardItem(props) {
  return (
    <>
      <li className='cards__item'>
        {props.href ? (
          <a className='cards__item__link' href={props.href} target="_blank" rel="noopener noreferrer">
            <figure className='cards__item__pic-wrap' data-category={props.label}>
              <img src={props.src} alt='Logo Image' className='cards__item__img' />
            </figure>
            <div className='cards__item__info'>
              <h5 className='cards__item__text'style={{ fontWeight: props.fontWeight || 'normal' }}>{props.text}</h5>
            </div>
          </a>
        ) : (
          <Link className='cards__item__link' to={props.path}>
            <figure className='cards__item__pic-wrap' data-category={props.label}>
              <img src={props.src} alt='Logo Image' className='cards__item__img' />
            </figure>
            <div className='cards__item__info'>
              <h5 className='cards__item__text' style={{ fontWeight: props.fontWeight || 'normal' }}>{props.text}</h5>
            </div>
          </Link>
        )}
      </li>
    </>
  );
}

export default CardItem;