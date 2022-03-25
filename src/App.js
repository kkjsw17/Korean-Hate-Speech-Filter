import { Box, Button, Grid, TextField, Alert, AlertTitle, Collapse } from '@mui/material';
import axios from 'axios';
import { useState } from 'react';
import banner from './banner.png';

const App = () => {
    const [comment, setComment] = useState('');
    const [errorOpen, setErrorOpen] = useState(false);
    const [successOpen, setSuccessOpen] = useState(false);
    
    const postComment = async () => {
        await axios.post('http://127.0.0.1:5000/',
            { comment: comment },
        ).then(res => {
            if (res.data === 1) {
                setErrorOpen(true);
                setTimeout(() => {
                    setErrorOpen(false);
                }, 3000);
            } else {
                setSuccessOpen(true);
                setTimeout(() => {
                    setSuccessOpen(false);
                }, 3000);
            }
        });

        setComment('');
    };

    return (
        <Box
            sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                mt: 5
            }}
        >
        <img src={banner} alt="banner" width='50%' />
        <Grid container spacing={0.5} width='40%'>
            <Grid item xs={10}>
                <TextField
                    multiline
                    fullWidth
                    value={comment}
                    label="Hate Speech Filter"
                    placeholder="댓글을 입력하세요."
                    rows={4}
                    onChange={(e) => {
                        setComment(e.target.value.split('\n').join(''));
                    }}
                    onKeyDown={(e) => {
                        if (e.key === "Enter") {
                            postComment();
                        }
                    }}
                />
            </Grid>
            <Grid item xs={2}>
                <Button
                    variant='contained'
                    fullWidth
                    sx={{
                        height: '100%',
                        fontSize: 18,
                        background: 'linear-gradient(135deg, #a91e24 25% ,#192156)'
                    }}
                    onClick={postComment}
                >
                    작성
                </Button>
            </Grid>
        </Grid>
        <Collapse
            in={errorOpen}
            sx={{
                position: 'fixed',
                top: 10,
                left: 10,
                right: 15
            }}
        >
            <Alert
                severity="error"
                fullWidth
            >
                <AlertTitle>악플 감지</AlertTitle>
                적절하지 않은 댓글입니다. 다시 입력해주세요. — <strong>BERT가 올바른 댓글 문화를 선도해나갑니다!</strong>
            </Alert>  
        </Collapse>
        <Collapse
            in={successOpen}
            sx={{
                position: 'fixed',
                top: 10,
                left: 10,
                right: 15
            }}
        >
            <Alert
                severity="success"
                width="100%"
            >
                <AlertTitle>댓글 저장</AlertTitle>
                입력하신 댓글이 저장되었습니다. — <strong>BERT가 올바른 댓글 문화를 선도해나갑니다!</strong>
            </Alert>
        </Collapse>
    </Box>
  );
}

export default App;
