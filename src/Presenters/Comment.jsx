const Comment = (props) => {
    return (
        <div>
            {
                props.comments.map((comment) => {
                    return(
                        <div>
                            {comment.date}
                            {comment.writer}
                            {comment.content}
                        </div>
                    )
                })
            }
        </div>
    );
}

export default Comment;