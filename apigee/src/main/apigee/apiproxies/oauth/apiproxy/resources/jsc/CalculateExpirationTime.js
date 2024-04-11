var firebase_id_time = context.getVariable('jwt.VerifyJWT-FirebaseIDToken.decoded.claim.exp') * 1000;
var curr_time = context.getVariable('client.received.end.timestamp');
context.setVariable("token_expiry", (firebase_id_time - curr_time).toString());