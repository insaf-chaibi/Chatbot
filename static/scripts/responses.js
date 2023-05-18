async function getBotResponse(input) {

    //console.log(input)
    let res=await fetch("http://127.0.0.1:8000/chatbot",{
        method: 'POST',
        body: JSON.stringify(input),
        mode: 'cors', // Resource sharing
        headers: {
        'Content-Type': 'application/json'
        },
    })
    let final = await res.json();
    return final['result'];
    
}