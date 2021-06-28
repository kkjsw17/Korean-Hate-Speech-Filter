import style from '../resources/css/home.module.css'
import Comment from '../Presenters/Comment'
import axios from 'axios'

const Home = () => {
    const comments = [];

    const fetchComment = async () => {
        var input = document.getElementById('input');
        var btn = document.getElementById('btn');
        var check;

        btn.disabled = true; input.disabled = true;
        input.style.backgroundColor = '#E0E0E0';

        await axios.post('api',
            {comment: input.value})
        .then((response) => {
            console.log(response.data);
            check = Number(response.data);
            if (check)
                alert("적절하지 않은 댓글입니다. 다시 입력해주세요.");
            else
                alert("입력하신 댓글이 저장되었습니다.");
        }).catch((error) => {
            console.log(error, 'error');
            alert(error);
        });

        if (!check)
            addComment(input.value);
        input.value = "";
        btn.disabled = false; input.disabled = false;
        input.style.backgroundColor = 'white';
    }

    const enterkey = () => {
        if (window.event.keyCode === 13 && !document.getElementById('input').disabled) {
            fetchComment();
        }
    }

    const addComment = (text) => {
        comments.push({
            uuid: comments.length + 1,
            content: text,
            writer: '권기준',
            date: new Date().toISOString().slice(0, 10)
        });
        console.log(comments);
    }

    return(
        <div className={style.home}>
            <Comment id='comments' comments={comments}></Comment>
            <textarea id="input" name='input' placeholder="댓글을 입력하세요." onKeyDown={enterkey}></textarea>
            <button type="submit" id="btn" onClick={fetchComment}>등록</button>
        </div>
    );
}

export default Home;