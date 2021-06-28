const express = require('express');
const router = express();
const spawnSync = require('child_process').spawnSync;

router.post('/', (req, res) => {
    console.log(req.body.comment);
    const python = spawnSync('python3', ['./src/predict.py', req.body.comment]);
    console.log(python.stdout.toString());
    res.send(python.stdout.toString());
});

module.exports = router;