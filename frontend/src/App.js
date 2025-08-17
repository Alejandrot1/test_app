import React, { useState } from 'react';

function App() {
    const [item, setItem] = useState({ name: '', description: '', price: 0, tax: 0 });

    const handleChange = (e) => {
        const { name, value } = e.target;
        setItem({ ...item, [name]: value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        const response = await fetch('http://localhost:8000/items/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(item),
        });
        const data = await response.json();
        console.log(data);
    };

    return (
        <div>
            <h1>Create Item</h1>
            <form onSubmit={handleSubmit}>
                <input name="name" placeholder="Name" onChange={handleChange} required />
                <input name="description" placeholder="Description" onChange={handleChange} />
                <input name="price" type="number" placeholder="Price" onChange={handleChange} required />
                <input name="tax" type="number" placeholder="Tax" onChange={handleChange} />
                <button type="submit">Submit</button>
            </form>
        </div>
    );
}

export default App;