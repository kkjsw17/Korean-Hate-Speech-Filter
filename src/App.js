import { Box, Button, Grid, TextField } from '@mui/material';
import axios from 'axios';
import { useState } from 'react';

const App = () => {
  const [comment, setComment] = useState('');
  
  const postComment = async () => {
    await axios.post('http://127.0.0.1:5000/',
      { comment: comment },
    ).then(res => {
      if (res.data === 1) {
        alert('악플');
      } else {
        alert('노악플');
      }
    });

    setComment('');
  };

  return (
    <Box>
      <Grid container spacing={0.5}>
        <Grid item xs={10}>
          <TextField
            multiline
            fullWidth
            value={comment}
            label="Hate Speech Filter"
            placeholder="댓글을 입력하세요."
            onChange={(e) => { setComment(e.target.value) }}
          />
        </Grid>
        <Grid item xs={2}>
          <Button
            variant='contained'
            fullWidth
            sx={{ height: '100%', fontSize: 18 }}
            onClick={postComment}
          >
            작성
          </Button>
        </Grid>
      </Grid>
    </Box>
  );
}

export default App;
